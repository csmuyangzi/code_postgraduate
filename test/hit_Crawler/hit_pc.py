# -*- coding:utf-8 -*-
import requests
import re
import csv
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import codecs
import sys
from selenium.webdriver.common.keys import Keys
reload(sys)
sys.setdefaultencoding('utf-8')

def pc_hit(text):
    driver = webdriver.PhantomJS(executable_path = '//Users/muyangzi/python/phantomjs-2.1.1-macosx/bin/phantomjs')
    driver.get("http://wi.hit.edu.cn/cemr/")
    time.sleep(3)
    source = driver.find_element_by_name('source')
    submit = driver.find_element_by_id('exectute')
    source.clear()
    time.sleep(3)
    source.send_keys(text)
    time.sleep(3)
    submit.click()

    pageSource = driver.page_source
    bsObj = BeautifulSoup(pageSource,'html.parser')

    fenci = bsObj.find('div',{'id':'segment'}).findAll('p')
    entity = bsObj.find('div',{'id':'entity'}).findAll('span')
    relation = bsObj.find('div',{'id':'relation'})

    list_fenci = [u"分词".encode("utf-8")]
    list_relation = [u"关系".encode('utf-8')]
    jibing =[u"疾病".encode("utf-8")];zhenduan=[u"诊断".encode("utf-8")];zisu=[u"自诉".encode("utf-8")]
    yichangjiancha=[u"异常检查".encode("utf-8")];jiancha=[u"检查".encode("utf-8")];zhiliao=[u"治疗".encode("utf-8")]

    for tmp in fenci:
        list_fenci.append(tmp.get_text())
    for tmp2 in relation:
        list_relation.append(tmp2.get_text())

    for cell in entity:
        if cell["style"] == "color:#ff0000;":
            jibing.append(cell.get_text())
        elif cell["style"] == "color:#ff00a0;":
            zhenduan.append(cell.get_text())
        elif cell["style"] == "color:#0000ff;":
            zisu.append(cell.get_text())
        elif cell["style"] == "color:#00b8ff;":
            yichangjiancha.append(cell.get_text())
        elif cell["style"] == "color:#B8860B;":
            jiancha.append(cell.get_text())
        else:
            zhiliao.append(cell.get_text())

    csvFile = open("hti_test.csv", 'wb')
    csvFile.write(codecs.BOM_UTF8)
    writer = csv.writer(csvFile,dialect='excel')

    writer.writerow(list_fenci)
    writer.writerow(jibing)
    writer.writerow(zhenduan)
    writer.writerow(zisu)
    writer.writerow(yichangjiancha)
    writer.writerow(jiancha)
    writer.writerow(zhiliao)
    writer.writerow(list_relation)
    csvFile.close()

    #return str_fenci,jibing,zhenduan,zisu,yichangjiancha,jiancha,zhiliao,list_relation



if __name__ == '__main__':
    text = u"""患者于患者四十余年以来每因季节变化均出现咳嗽、咯痰，近年来上述症状发作后伴有活动性气促，
6小时余前无明显诱因出现意识不清，无口吐白沫，无肢体偏瘫，由保姆发现，拨打120送至海南省人民医院急诊科就诊，测血压最低60/40mmhg，
测体温最高39℃，做CT提示：肺部感染，诊断：1.肺性脑病2.重症肺炎，治疗上予持续泵入去甲肾上腺素注射液0.3μg/min升血压，
插管接有创呼吸机保证呼吸，患者家属要求到我院治疗，由急救车送至我院急诊，以“1.肺性脑病2.重症肺炎”收入我科。
入院症见：浅昏迷，留置气管插管，可吸出少量黄色粘痰。"""
    pc_hit(text)