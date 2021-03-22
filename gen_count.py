import smtplib
import ssl
import pandas as pd
import os
from os import path
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from optparse import OptionParser
import pytz
import psycopg2

TH_PG_HOST = 'dipperprodnew-truck-histories-replica.canbbkmz75pp.ap-south-1.rds.amazonaws.com'
TH_PG_PORT = '5432'
TH_PG_DATABASE = 'gps_development_postgres'
TH_PG_USERNAME = 'ec2-user'
TH_PG_PASSWORD = 'tester'

MAIL_ADDRESS = ["praveen.jha@getfareye.com", "sandeep.bindra@getfareye.com", "nitish.gupta@getfareye.com",
                "b.bhopalwala@getfareye.com", "ayush.syal@getfareye.com", "vaibhav.gupta@getfareye.com", "ayush.singh@getfareye.com"]

DIPPER_SMTP_ADDRESS = "email-smtp.us-east-1.amazonaws.com"
DIPPER_SMTP_PORT = 465
DIPPER_SMTP_DOMAIN = "portal.usedipper.com"
DIPPER_SMTP_USERNAME = "AKIAI2NXE2WJRPNIAGDQ"
DIPPER_SMTP_PASS = "AlOOoZE0U4KpVtd3WPK+HKZw/84d/UjlN+qhsiqt1PEA"

connection = psycopg2.connect(user=TH_PG_USERNAME, password=TH_PG_PASSWORD, host=TH_PG_HOST, port=TH_PG_PORT,
                              database=TH_PG_DATABASE)

flag_vars = [0, 3, 4, 5]

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FMT = '%Y-%m-%d'

timezone = pytz.timezone("Asia/Kolkata")




def generate_records(start_date, end_date):
    record_flags = []
    for item in flag_vars:
        sql_query = "select count(1) from truck_histories th where th.ist_timestamp >='{start_date}' and th.ist_timestamp <= '{end_date}' and is_processed = {process_status};" \
            .format(start_date=start_date, end_date=end_date, process_status=str(item))
        cursor = connection.cursor()
        cursor.execute(sql_query)
        record = cursor.fetchone()
        temp = dict(processing_state=item, count=record[0])
        record_flags.append(temp)
    return record_flags

#
# """
# ========== Writing Processing State Records to file ============
# """


# with open(file_name, 'w') as write:
#     json.dump(record_flags, write)


def send_email_with_attachment(file_name, subject):
    subject = subject
    receiver_email = MAIL_ADDRESS
    sender_email = "report@" + DIPPER_SMTP_DOMAIN

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = "report@" + DIPPER_SMTP_DOMAIN
    message["To"] = ','.join(MAIL_ADDRESS)
    message["Subject"] = subject

    with open(file_name, "rb") as attached_file:
        part = MIMEApplication(
            attached_file.read(),
            Name=path.basename(file_name)
        )
    part['Content-Disposition'] = 'attachment; filename="%s"' % path.basename(file_name)

    message.attach(part)
    message_text = message.as_string()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(DIPPER_SMTP_ADDRESS, DIPPER_SMTP_PORT, context=context) as server:
        server.login(DIPPER_SMTP_USERNAME, DIPPER_SMTP_PASS)
        server.sendmail(sender_email, receiver_email, message_text)


def alert_creator(type):
    now = datetime.now(timezone)
    # Ist Now Time
    now_ist = now.strftime(format=DATE_FORMAT)
    file_name = now_ist + '-processing_state.csv'
    unprocessed_pings = [0, 3, 4]

    unprocessed_count = 0
    processed_count = 0


    if type == 'hourly':
        one_hour = now - timedelta(hours=1)
        # Ist Timezone for one hour before
        #one_hour_ist = timezone.localize(one_hour)

        one_hour_ist = one_hour.strftime(format=DATE_FORMAT)
        record_flags = generate_records(one_hour_ist, now_ist)

        ## Logic to Check the Count
        for process in record_flags:
            if process['processing_state'] in unprocessed_pings:
                unprocessed_count += process['count']
            else:
                processed_count += process['count']
        total_count = (unprocessed_count + processed_count)

        unprocessing_percentage = (unprocessed_count / total_count) * 100
        if unprocessing_percentage > 5:
            pd.DataFrame(record_flags).to_csv(file_name, index=False)
            subject = "Hourly Report - unprocessed percentage is greater than 5%"
            send_email_with_attachment(file_name, subject)
            os.remove(file_name)
        elif total_count < 400000:
            pd.DataFrame(record_flags).to_csv(file_name, index=False)
            subject = "Hourly Report - total count is less than expected"
            send_email_with_attachment(file_name, subject)
            os.remove(file_name)
    else:
        start_date = now.strftime(format=DATE_FMT) + ' 00:00:00'
        end_date = now.strftime(format=DATE_FMT) + ' 23:59:00'
        record_flags = generate_records(start_date, end_date)
        pd.DataFrame(record_flags).to_csv(file_name, index=False)
        subject = "Daily summary of pings processing distribution"
        send_email_with_attachment(file_name, subject)
        os.remove(file_name)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--type_val", dest="type_val", help="Type of alerts that is run", metavar="type_val")
    (options, args) = parser.parse_args()
    alert_creator(options.type_val)
