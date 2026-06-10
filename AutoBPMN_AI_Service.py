import numpy as np
import Threshold_Strategies as ts
import Background as bg
import Further_Dimension_Approaches as fd

import New_And_State_of_the_art_Embeddings as emb

STRATEGY = 1  #default threshold strategy (Best performance in Evaluation)

def process(body):
    similarity_panel=body.get("similarity_panel", False)
    text=body.get("text")
    bpmn_xml=body.get("bpmn_xml")
    LEMMATIZE=body.get("lemmatize", False)
    REMOVE_COND=body.get("remove_cond", False)
    
    if text is None or bpmn_xml is None:
        raise ValueError("'text' and 'bpmn_xml' must be provided in the request body")

    #If similarity_panel is False, skip all approaches and return only the match-based F1
    if not similarity_panel:
        return compute_basic_f1(text, bpmn_xml)

    approach = body.get("approach", "model2text")
    methods = body.get("methods", [])

    if not methods:
        raise ValueError("No methods provided in 'methods' list.")

    if approach!="consensus" and len(methods)>1:
        raise ValueError(f"Multiple methods are only allowed for 'consensus' matching. Approach '{approach}' requires exactly 1 method.")

    if approach=="consensus" and len(methods)<2:
        raise ValueError(f"Approach 'consensus' requires at least 2 methods.")
    cfg=methods[0]
    
    # Consensus
    if approach=="consensus":
        consensus_sim,sentences, tasks, f1, labels=fd.consensus_matching("custom", methods, text=text, bpmn_xml=bpmn_xml, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
        threshold=int(len(methods)*2/3)/len(methods)
        return [result(f"Consensus ({', '.join(labels)})", consensus_sim, threshold, sentences, tasks)]


    data_dict = ts.load_data(cfg, ["custom"], LEMMATIZE, REMOVE_COND, text=text, bpmn_xml=bpmn_xml)
    data = data_dict["custom"]
    sentences = data["sentences"]
    tasks = data["tasks"]

    #Tuple Matching
    if approach=="tuple":
        sim, _, _,s_ranges, t_ranges=fd.tuple_matching(data,cfg)
        t=ts.get_precomputed_threshold(cfg, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
        tuple_sentences=[]
        for r in s_ranges: 
            temp_s = []
            for g in range(r[0], r[1]+1):
                temp_s.append(f"S{g+1}: {sentences[g]}")
            s_str = ". ".join(temp_s)
            tuple_sentences.append(s_str)
        tuple_tasks = []
        for r in t_ranges:
            temp_t = []
            for g in range(r[0], r[1]+1):
                temp_t.append(f"T{g+1}: {tasks[g]}")
            t_str = ". ".join(temp_t)
            tuple_tasks.append(t_str)
        return [result("Tuple Matching",sim,t,tuple_sentences,tuple_tasks)]

    #Best-Of-Tuple
    if approach=="best_of_tuple":
        sim,combos=fd.best_of_tuple_matching(data,cfg)
        t=ts.get_precomputed_threshold(cfg, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND) #######################
        bot_tasks=[]
        for grp in combos:
            temp_b = []
            for g in grp:
                temp_b.append(f"T{g+1}: {tasks[g]}")
            b_str = ". ".join(temp_b)
            bot_tasks.append(b_str)
        return [result("Best-Of-Tuple",sim,t,sentences,bot_tasks)]

    #Default Model2text
    results=[]
    for m_cfg in methods:
        sim = ts.get_sim_matrix(data_dict["custom"], m_cfg)
        if "embedding" in m_cfg:
            label = f"{m_cfg['embedding'].upper()} + {m_cfg['metric'].upper()}"
        else:
            label = m_cfg["traditional"].upper()
        t = ts.get_precomputed_threshold(m_cfg, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
        results.append(result(label, sim, t, sentences, tasks))
    return results


def compute_basic_f1(text, bpmn_xml):
    LEMMATIZE=True #set permantly  ###############################
    REMOVE_COND=True#set permantly ##################################
    cfg = {"embedding": "bert", "metric": "cos"}
    data_dict = ts.load_data(cfg, ["custom"], LEMMATIZE, REMOVE_COND, text=text, bpmn_xml=bpmn_xml)
    data = data_dict["custom"]
    sim = ts.get_sim_matrix(data, cfg)
    threshold = ts.get_precomputed_threshold(cfg, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    f1 = bg.compute_match_f1(sim, threshold)
    return {
        "f1": round(float(f1), 2),
        "threshold": round(float(threshold), 2),
    }


def result(label, sim, threshold, sentences, tasks):
    f1 = bg.compute_match_f1(sim, threshold)
    return {
        "label": label,
        "sim_matrix": sim.tolist(), 
        "f1": round(float(f1), 2),
        "threshold": round(float(threshold), 2),
        "sentences": sentences,
        "tasks": tasks
    }


    


if __name__ == "__main__":
    TEXT = "The customer places an order. We receive the order and process the payment. If you confirm the customer order, the goods are shipped to the customer."
    
    BPMN_XML = """<testset xmlns="http://cpee.org/ns/properties/2.0">
  <description>
    <description xmlns="http://cpee.org/ns/description/1.0">
      <call id="a1" endpoint="auto">
        <parameters>
          <label>Receive customer order</label>
        </parameters>
      </call>
      <call id="a2" endpoint="auto">
        <parameters>
          <label>Check inventory</label>
        </parameters>
      </call>
      <call id="a3" endpoint="auto">
        <parameters>
          <label>Process payment</label>
        </parameters>
      </call>
      <call id="a4" endpoint="auto">
        <parameters>
          <label>Ship goods</label>
        </parameters>
      </call>
    </description>
  </description>
</testset>"""
 
    request_body = {
        "similarity_panel": False,
        "text": TEXT,
        "bpmn_xml": BPMN_XML,
        "approach": "model2text",
        "methods": [
            {"embedding": "bert", "metric": "cos"},
        ],
        "lemmatize": True,
        "remove_cond": True,
           
    }
    
    results = process(request_body)



    if request_body.get("similarity_panel"):
        print("---Res---")
        for res in results:
            print("\nLab:", res['label'])
            print("f1:",res['f1'])
            print("t:",res['threshold'])
            print("Sentences:")
            for s in res['sentences']:
                print(f"{s}")
            print("Tasks:")
            for t in res['tasks']:
                print(f"{t}")
            
            print("Similarity Matrix:")
            matrix = res['sim_matrix']
            if not matrix:
                print("  []")
            else:
                for row in matrix:
                    print(f"{row}")

    else:
        print("f1:",results['f1'])
        print("t:",results['threshold'])
        
  

