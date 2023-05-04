from rasa.nlu.components import Component
from rasa.nlu import utils
from rasa.nlu.model import Metadata
import os
import spacy
import en_core_web_md
from rasa.shared.nlu.constants import TEXT
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
#nprint(TEXT)
global nlp
nlp = spacy.load('en_core_web_md')
class SpaceDetector(Component):
    """A pre-trained sentiment component"""
    name = "space"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["en"]


    def __init__(self, component_config=None):
        super(SpaceDetector, self).__init__(component_config)


    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass
    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""
        
        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "space",
                  "extractor": "space_extractor"}


        return entity
    def process(self, message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""
        global nlp
        #res = sid.polarity_scores(message.text)
        #key, value = max(res.items(), key=lambda x: x[1])
        
        l = [' Broadsign']
        sim = []
        #
        # print(str(message.build()))
        #print(str(message.get_full_intent))
        #print(message.text)
        #print(message.data)
        #print(str(message.data['text']))
        #print(message.get(TEXT))
        msg = nlp(str(message.get(TEXT)))
        msg1 = str(message.get(TEXT)).lower()
        if (msg1.find("broadsign") != -1):
            entity = self.convert_to_rasa(l[0], 0.99)
            message.set("entities", [entity], add_to_output=True)
        elif (msg1.find("outsystems") != -1):
            entity = self.convert_to_rasa(l[1], 0.99)
            message.set("entities", [entity], add_to_output=True)
        elif (msg1.find("chain") != -1):
            entity = self.convert_to_rasa(l[2], 0.99)
            message.set("entities", [entity], add_to_output=True)
        elif (msg1.find("support") != -1):
            entity = self.convert_to_rasa(l[3], 0.99)
            message.set("entities", [entity], add_to_output=True)
        else:
            for i in l:
                p = nlp(i)
                simi  = p.similarity(msg)
                sim.append(simi)
                print(simi,i)
            value = max(sim)
            key = l[sim.index(value)]
            entity = self.convert_to_rasa(key, value)
            if value >= 0.8:
                message.set("entities", [entity], add_to_output=True)

    def persist(self,file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""
        pass