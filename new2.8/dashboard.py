from rasa.nlu.components import Component
from rasa.nlu import utils
from rasa.nlu.model import Metadata
import os
import spacy
import en_core_web_md
from rasa.shared.nlu.constants import TEXT
from actions.actions import ActionHelloWorld

#print(TEXT)
global nlp
nlp = spacy.load('en_core_web_md')
class DashboardDetector(Component):
    """A pre-trained sentiment component"""
    name = "dashboard"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["en"]


    def __init__(self, component_config=None):
        super(DashboardDetector, self).__init__(component_config)


    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass
    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""
        
        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "dashboard",
                  "extractor": "dashboard_extractor"}


        return entity
    def process(self, message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""
        global nlp
        #res = sid.polarity_scores(message.text)
        #key, value = max(res.items(), key=lambda x: x[1])
        f = open("./actions/space.txt","r")
        space = str(f.read())
        f.close()

        # f = open("./actions/space.txt","w")
        # f.write("")
        # f.close()

        #print(1111,space)
        l = {"Broadsign":
            
            ["reporting dashboard"]
        }
        if space != "":
            l = l[space]
        

            # ]
            #print(l)
            sim = []
        
            msg = nlp(str(message.get(TEXT)))
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