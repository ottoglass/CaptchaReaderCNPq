import json
from datetime import datetime
from io import BytesIO

import bs4
import numpy as np
import requests
from matplotlib import pyplot as plt
from PIL import Image

import cv2
from kerasNN import Network
from web_forms import *


class CNPqLattes:

    def __init__(self):
        self.s = requests.Session()
        self.NN = Network()
        self.NN.load("Models\\Model-Arch3-3")

    def __del__(self):
        self.s.close()

    def requestCV(self, ID):
        r = self.s.get(
            "http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id="+ID)
        return r

    def requestCaptcha(self):
        now = int(datetime.now().timestamp())
        r = self.s.get(
            "http://buscatextual.cnpq.br/buscatextual/servlet/captcha?metodo=getImagemCaptcha&noCache="+str(now))
        image = np.array(Image.open(BytesIO(r.content)))
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def validCaptcha(self, ID, captcha):
        captcha_form["id"] = ID
        captcha_form["informado"] = captcha
        captcha_form["metodo"] = "validaCaptcha"
        r = self.s.get(
            "http://buscatextual.cnpq.br/buscatextual/servlet/captcha", params=captcha_form)
        response = json.loads(r.text)
        if response["estado"] == "sucesso":
            return True
        else:
            return False

    def findIDs(self, soups):
        soups = soups.findAll("b")
        IDs = []
        for soup in soups:
            if (soup.find("a") == None):
                continue
            link = soup.a["href"]
            id_lower_index = link.find("'")+1
            id_upper_index = link.find("'", id_lower_index)
            IDs.append(link[id_lower_index:id_upper_index])
        return IDs

    def search(self, search_string, num_items=10, buscar_demais=False):
        form = search_form_post.copy()
        form["textoBusca"] = search_string
        form["buscarDemais"] = buscar_demais
        response = self.s.post(
            "http://buscatextual.cnpq.br/buscatextual/busca.do", data=form)
        if response.status_code == 404:
            print("Error 404")
            return []
        response_soup = bs4.BeautifulSoup(response.text, "html.parser")
        query_string = response_soup.find(
            'input', {'name': 'query'}).get('value')
        num_results = int(response_soup.find('b').text)
        form = search_form_page_foward.copy()
        form["query"] = query_string
        max_results = str(min(num_results, num_items))
        form["registros"] += max_results
        response = self.s.get(
            "http://buscatextual.cnpq.br/buscatextual/busca.do", params=form)
        response_soup = bs4.BeautifulSoup(response.text, "html.parser")
        IDs = self.findIDs(response_soup)
        search_results = []
        print("Fetching "+max_results+" Items")
        for ID_idx, ID in enumerate(IDs):
            print(str(ID_idx+1)+"/"+max_results, end="\r")
            while True:
                captcha = self.NN.predictCaptcha(self.requestCaptcha())
                if self.validCaptcha(ID, captcha):
                    break
            r = self.requestCV(ID)
            response_soup = bs4.BeautifulSoup(r.text, "html.parser")
            result = {"Nome": response_soup.select(".nome")[0].text,
                      "Resumo": response_soup.select(".resumo")[0].text}
            search_results.append(result)
        print("Done!")
        return search_results
