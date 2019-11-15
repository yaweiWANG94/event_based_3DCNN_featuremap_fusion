# -*- coding: utf-8 -*-
"""
根据xml生成txt文件
@author: bai
"""
import sys
import os
import xml.etree.ElementTree as ET
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='xml2txt demo')
    parser.add_argument('--txt','-T', dest='txt_path', help='txt path to save',
                        default='None')
    parser.add_argument('--Annotations','-A', dest='xml_path', help='xml path to read',
                        default='None')
    args = parser.parse_args()

    return args

args=parse_args()

path_to_folder = args.xml_path
if path_to_folder[-1] != '/':
    path_to_folder += '/'
# create VOC format files
xml_list = os.listdir(path_to_folder)

if len(xml_list) == 0:
  print("Error: no .xml files found in ground-truth")
  sys.exit()
for tmp_file in xml_list:
  #print(tmp_file)
  # 1. create new file (VOC format)
  #print(tmp_file[13:-4])
  #with open(os.path.join(args.txt_path,'I'+"{:0>6d}".format(int(tmp_file[6:-4]))+".txt"), "w") as new_f:
  with open(os.path.join(args.txt_path,tmp_file.replace(".xml", ".txt")), "w") as new_f:
    xmlpath=os.path.join(args.xml_path,tmp_file)
    root = ET.parse(xmlpath).getroot()
    for obj in root.findall('object'):
      obj_name = obj.find('name').text
      bndbox = obj.find('bndbox')
      left = bndbox.find('xmin').text
      top = bndbox.find('ymin').text
      right = bndbox.find('xmax').text
      bottom = bndbox.find('ymax').text
      new_f.write('1'+ " " + left + " " + top + " " + right + " " + bottom +'\n')
print("Conversion completed!")
