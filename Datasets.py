import xml.etree.ElementTree as ET
import os
import numpy as np
import spacy
import re

test_cpee= """<testset>
<executionhandler>ruby</executionhandler>
<dataelements/>
<endpoints>
<user>https-post://cpee.org/services/timeout-user.php</user>
<auto>https-post://cpee.org/services/timeout-auto.php</auto>
<subprocess>https-post://cpee.org/flow/start/url/</subprocess>
<timeout>https-post://cpee.org/services/timeout.php</timeout>
<send>
https-post://cpee.org/ing/correlators/message/send/
</send>
<receive>
https-get://cpee.org/ing/correlators/message/receive/
</receive>
</endpoints>
<attributes>
<info>Plain Instance</info>
<modeltype>CPEE</modeltype>
<theme>preset</theme>
</attributes>
<description>
<description>
<call id="a1" endpoint="" a:alt_id="1">
<parameters>
<label>Transmit Transaction Data Request</label>
<method/>
<type>:task</type>
<arguments/>
</parameters>
</call>
<call id="a3" endpoint="" a:alt_id="2">
<parameters>
<label>Check Request</label>
<method/>
<type>:task</type>
<arguments/>
</parameters>
</call>
<choose mode="exclusive" a:alt_id="3">
<alternative condition="Data Transmission" language="text/javascript">
<call id="a7" endpoint="" a:alt_id="4">
<parameters>
<label>Transmit Data</label>
<method/>
<type>:task</type>
<arguments/>
</parameters>
</call>
</alternative>
<alternative condition="Rejection" language="text/javascript">
<call id="a9" endpoint="" a:alt_id="5">
<parameters>
<label>Reject Request</label>
<method/>
<type>:task</type>
<arguments/>
</parameters>
</call>
</alternative>
</choose>
</description>
</description>
<transformation>
<description type="copy"/>
<dataelements type="none"/>
<endpoints type="none"/>
</transformation>
</testset>"""

def task_extraction(filepath_or_xml):
    #detrmine if the input is a XML string or a file path
    if isinstance(filepath_or_xml, str) and filepath_or_xml.strip().startswith("<"):
        xml_content = filepath_or_xml
    else:
        #load XML content from the file path
        with open(filepath_or_xml, "r", encoding="utf-8") as f:
            xml_content = f.read()

    #remove attributes with prefixes (like a:alt_id="1") to avoid prefix errors (regex generated through Gemini-3.5-Flash to shorten the code, but was extensively tested for correctness)
    #Prompt: "Write a regex for Python's re.sub to remove all XML attributes that have a namespace prefix (like a:alt_id="1" or xmlns:xsi="..."). It should remove the attribute name, the equals sign, and the quoted value, including the leading whitespace."
    xml_content = re.sub(r'\s+[a-zA-Z_][\w.-]*:[a-zA-Z_][\w.-]*\s*=\s*(?:"[^"]*"|\'[^\']*\')', '', xml_content)
    root = ET.fromstring(xml_content)

    task_labels = []

      #call tags from CPEE XMLs
    for node in root.iter():
        if node.tag.endswith("call"):
            for child in node.iter():
                if child.tag.endswith("label") and child.text:
                    task_labels.append(child.text.strip().lower())
                    break # only take the first label under this call
    
   # activity tags from PMo XMLs
    if not task_labels:
        for node in root.iter():
            if node.tag.endswith("activity"):
                label = node.get("action")
                if label:
                    task_labels.append(label.strip().lower())

    return task_labels


def sentence_extraction(text):
  
    #split text according to specific symbols
    sentences = []
    temp_text = text.replace("e.g.", "§EG§").replace("etc.","§ETC§").replace("i.e.","§IE§")  #some symbols (e.g., "etc.") should not split the sentence
    sentence = ""
    for char in temp_text:
        if char in ".!?;":
            sentences.append(sentence.strip())
            sentence = ""
        elif char == "\n":
            sentence += " "  #replace paragraphs with Space
        else:
            sentence += char
    replaced_sentences = []
    for s in sentences:
        replaced_sentences.append(s.replace("§EG§", "e.g.").replace("§ETC§", "etc.").replace("§IE§", "i.e."))
    sentences = replaced_sentences

    # remove empty strings
    final_sentences=[]
    for s in sentences:
        if s.strip():
            final_sentences.append(s.strip().lower())
    sentences=final_sentences

    return sentences

nlp = spacy.load("en_core_web_sm")
def lemmatizer(tasks, sentences): # https://spacy.io/usage/linguistic-features
    lemmatized_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        lemmas = []
        for token in doc:
            lemmas.append(token.lemma_)
        lemmatized_sentences.append(" ".join(lemmas).replace(" ,", ","))

    lemmatized_tasks = []
    for task in tasks:
        doc = nlp(task)
        lemmas = []
        for token in doc:
            lemmas.append(token.lemma_)
        lemmatized_tasks.append(" ".join(lemmas).replace(" ,", ","))

    return lemmatized_tasks, lemmatized_sentences


def remove_conditional_statements(sentences):
    cleaned_sentences = []
    
    # Regex used to shorten the code(partially generated by Gemini-3.5-Flash to correct regex syntax errors, but extensively tested for correctness)
    #prompt: "I need two regular expressions to clean up sentences by stripping out conditional statements. The trigger words are: If, Once, When, In case, After, Following, Upon. The first regex should detect conditions at the very beginning of the string and match everything up to the first comma so I can delete it. The second regex should detect conditions appended later in the sentence, matching from a comma followed by a trigger word, up to the next comma."
    start_pattern = re.compile(r'^(?:If|Once|When|In case|After|Following|Upon)\b[^,]*,', re.IGNORECASE)
    mid_pattern = re.compile(r',\s*\b(?:if|once|when|in case|after|following|upon)\b[^,]*', re.IGNORECASE)
    
    for text in sentences:
        
        #check if the condition is at the beginning of the sentence
        if start_pattern.search(text):
            #Remove the matched part (including the comma)
            text = start_pattern.sub('', text, count=1)
        else:
            # check if the condition is in the middle/end
            text = mid_pattern.sub('', text, count=1)
            
       #replaces any multiple spaces with a single space
        text = " ".join(text.split())
        
        cleaned_sentences.append(text)
            
    return cleaned_sentences

#load existing doc_ids
def load_dataset(doc_id, lemmatized=False,remove_conditions=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    id_num = int(doc_id)
    if 1 <= id_num <= 10:
        subfolder = "training_data"
    elif 11 <= id_num <= 21:
        subfolder = "validation_data"
    else:
        subfolder = "validation_data"

    bpmn_path = os.path.join(base_dir, "Datasets", "bpmn_text", subfolder, f"{doc_id}.xml")
    text_path = os.path.join(base_dir, "Datasets", "descriptions", subfolder, f"{doc_id}.txt")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tasks = task_extraction(bpmn_path)
    sentences = sentence_extraction(text)

    if remove_conditions:
        sentences = remove_conditional_statements(sentences)
    if lemmatized:
        return lemmatizer(tasks, sentences)

    return tasks, sentences

#load custom data (important for AutoBPMN.AI for custom input)
def get_data(doc_id=None, text=None, bpmn_xml=None, lemmatized=False, remove_conditions=False):
    # Load data either from a doc_id OR from directly provided text + bpmn_xml.
    if text is not None and bpmn_xml is not None:
        tasks = task_extraction(bpmn_xml)
        sentences = sentence_extraction(text)
        if remove_conditions:
            sentences = remove_conditional_statements(sentences)
        if lemmatized:
            return lemmatizer(tasks, sentences)
        return tasks, sentences
    elif doc_id is not None:
        return load_dataset(doc_id, lemmatized, remove_conditions)


GROUND_TRUTH = {
    # training models 
    # Model 01 (6s x 8t)
    "01": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 1, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 1], 
        [0, 0, 0, 0, 0, 1, 0, 0]
    ]),
    # Model 02 (6s x 11t)
    "02": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]),
    # Model 03 (7s x 11t)
    "03": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ]),
    # Model 04 (7s x 10t)
    "04": np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ]),
    # Model 05 (6s x 12t)
    "05": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ]),
    # Model 06 (6s x 12t)
    "06": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    ]),
    # Model 07 (6s x 10t)
    "07": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]),
    # Model 08 (6s x 9t)
    "08": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 1, 0, 1, 0, 0],  
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]),
    # Model 09 (5s x 10t)
    "09": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ]),
    # Model 10 (19s x 25t)
    "10": np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    ]),
    
    #Validation models
    # Model 11 (21s x 26t)
    "11": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ]),
    # Model 12 (8s x 11t)
    "12": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    # Model 13 (9s x 15t)
    "13": np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ]),
    # Model 14 (10s x 15t) 
    "14": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]),
    # Model 15 (5s x 8t)
    "15": np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],  
        [0, 1, 1, 0, 0, 0, 0, 0],  
        [0, 0, 0, 1, 0, 0, 0, 0],  
        [0, 0, 0, 0, 1, 1, 0, 0],  
        [0, 0, 0, 0, 0, 0, 1, 1]
    ]),
    # Model 16 (3s x 4t) 
    "16": np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 1]
    ]),
    # Model 17 (3s x 4t)
    "17": np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 1]
    ]),
    # Model 18 (3s x 3t)
    "18": np.array([
        [1, 0, 0],  
        [0, 1, 0],  
        [0, 0, 1]
    ]),
    # Model 19 (5s x 5t) 
    "19": np.array([
        [1, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ]),
    # Model 20 (4s x 6t)
    "20": np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1]
    ]),
    # Model 21 (4s x 5t)
    "21": np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])
}

def get_ground_truth(doc_id):
    return GROUND_TRUTH.get(doc_id)

if __name__ == "__main__":
    """
    doc_id = "01" 
    print(f"Test Lemmatizer with Dataset {doc_id}")
    
    print("\n1. Without Condition-Removal:")
    tasks, sentences = load_dataset(doc_id, lemmatized=False, remove_conditions=False)
    for s in sentences[:5]:
        print(" -", s)
        
    print("\n2. With Condition-Removal")
    tasks_rm, sentences_rm = load_dataset(doc_id, lemmatized=False, remove_conditions=True)
    for s in sentences_rm[:5]:
        print(" -", s)
      """  
    
    doc_id = "11"
    LEMMATIZE=True
    REMOVE_COND=True
    tasks, sentences= load_dataset(doc_id, lemmatized=LEMMATIZE, remove_conditions=REMOVE_COND)
    print(f"sentences({len(sentences)}):")
    for i, s in enumerate(sentences, 1):
        print(f"Sentence {i}: {s}")

    print(f"\ntasks({len(tasks)}):")
    for i, t in enumerate(tasks, 1):
        print(f"Task {i}: {t}")
    

    
