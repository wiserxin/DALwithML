# !/usr/bin/env python
# -*- coding: utf-8 -*-
import smtplib
import email.mime.multipart
import email.mime.text
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import email
import os
import argparse


class send_mail:
    def __init__(self, From, To, Cc, pw, file_path, file_header, file_body):
        # 发送人
        self.From = From
        # 收件人['aaa@a.com','bbb@a.com']
        self.To = To
        # 抄送人
        self.Cc = list(Cc)
        # 登录邮件密码base64.encodestring('明文')加密后密码
        self.pw = pw
        # 文件具体路径(路径+文件名称)
        self.file_path = file_path
        # 标题头
        self.file_header = file_header
        # 内容
        self.file_body = file_body

    def login(self):
        mail_host = "smtp.{}".format(self.From.split('@')[1])
        server = smtplib.SMTP(mail_host)
        server.connect(mail_host, 25)
        server.starttls()
        server.login(self.From, self.pw)
        try:
            receive = self.To
            # receive.extend(self.Cc)
            server.sendmail(self.From, self.To, self.atta())
        except Exception as e:
            print(e)
            print("???")
        finally:
            server.quit()

    def atta(self):
        main_msg = MIMEMultipart()
        # 内容
        text_msg = MIMEText(self.file_body)
        main_msg.attach(text_msg)
        try:
            contype = 'application/octet-stream'
            maintype, subtype = contype.split('/', 1)

            data = open(self.file_path, 'rb')
            file_msg = MIMEBase(maintype, subtype)
            file_msg.set_payload(data.read())
            data.close()
            email.encoders.encode_base64(file_msg)
            basename = os.path.basename(self.file_path.split('/')[-1])
            file_msg.add_header('Content-Disposition', 'attachment', filename=basename)
            main_msg.attach(file_msg)
        except Exception as e:
            # print(e)
            print('without files .')
            pass

        main_msg['From'] = self.From
        main_msg['To'] = ";".join(self.To)
        main_msg['Cc'] = ";".join(self.Cc)

        # 标题头
        main_msg['Subject'] = self.file_header
        main_msg['Date'] = email.utils.formatdate()

        fullText = main_msg.as_string()

        return fullText


parser = argparse.ArgumentParser()

parser.add_argument('--From', type=str, default='xxx', help="发件人")
parser.add_argument('--To', type=str, default='False', help="收件人,可为List")
parser.add_argument('--Cc', type=str, default='', help="抄送")
parser.add_argument('--pw', type=str, default='', help="密码")
parser.add_argument('--file_path', type=str, default='', help="附件路径")
parser.add_argument('--header', type=str, default='ezEmail auto send', help="主题")
parser.add_argument('--body', type=str, default='from wiserxin ~', help="邮件内容")

args = parser.parse_args()

if __name__ == '__main__':
    # print(args)
    s = send_mail(args.From, [args.To], args.Cc, args.pw, args.file_path, args.header, args.body)
    s.login()
    print('发送成功！')
