# -*- coding: utf-8 -*-
"""
Created on Mon May 21 21:59:24 2022

@author: ulind
"""

import shutil, os
from keybert import KeyBERT
from google_images_download import google_images_download
import numpy as np
import skvideo.io
import cv2
from transformers import pipeline

class generateVideo:
    def __init__(self,article):
        self.article = article

    def summarize(self):
        '''Summarizes the article'''
        summarizer = pipeline("summarization")
        text = summarizer(self.article, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        return text

    def dowloadImages(self,doc):
        '''Downloads images based on keywords and stores them in downloads folder and returns a list of key words and sentences'''
        kw_model = KeyBERT()
        response = google_images_download.googleimagesdownload()
        sentences = doc.split('.')
        key_word_list = []
        for sentence in sentences:
            keyword = kw_model.extract_keywords(sentence,keyphrase_ngram_range=(2, 2),top_n=1)
            if len(keyword)>0:
                arguments = {"keywords":keyword[0][0],"limit":1}
                paths = response.download(arguments)
                key_word_list.append(keyword[0][0])
        return key_word_list,sentences


    def overlayText(self,img_name,doc):
        '''Overlays text on an image'''
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        try:
            img = cv2.resize(img, (800, 800))
        except:
            img = img[:800,:800,:]
        # Defining the text parameters
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 2
        lineType               = 2
        sentences = []
        # Splitting each sentence to multiple lines
        for cnt,letter in enumerate(doc):
            sentence = doc[:45]
            sentences.append(sentence)
            doc = doc[45:]
            if len(doc) < 45:
                sentences.append(doc)
                break
        # Creating a bottom border on the image for clear visibilty of overlayed text depending on the length of text
        border = 800 - 50*len(sentences)
        img[border:,:,:] = [125,125,125]
        # Overlaying text line by line in  an image
        for line_no in range(len(sentences)):
            start_x = 10 + 3*line_no
            start_y = 800 - 28*(len(sentences) - line_no)
            location = (start_x,start_y)
            img = cv2.putText(img,sentences[line_no],location,font, fontScale,fontColor,thickness,lineType)   
        return img

    def createVideo(self,doc):
        '''Stiches the images in the images folder and makes a video by overlaying appropriate text on top of them'''
        # Defining height and width of the video
        height = 800
        width = 800
        start = 0
        end = 125
        
        key_word_list,sentences = self.dowloadImages(doc)
        out_video =  np.empty([125*len(key_word_list), height, width, 3], dtype = np.uint8)
        out_video =  out_video.astype(np.uint8)

        for cnt,key_word in enumerate(key_word_list):
            os.chdir('/content/downloads'+'/'+key_word)
            for img_name in os.listdir():
                img = self.overlayText(img_name,sentences[cnt])
                out_video[start:end] = img
                start += 125
                end += 125
                
        # Writes the the output image sequences in a video file
        os.chdir('/content')
        skvideo.io.vwrite("video.mp4", out_video)
        shutil.rmtree('/content/downloads')
        print('Video created!!')
        
