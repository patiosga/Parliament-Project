import pandas as pd
import string
import networkx as nx
import matplotlib.pyplot as plt
from variables import stopwords
from preprocess_speech import preprocess_speech
import pandas as pd
import json


def load_file():
    file_path = r"processed_data.csv"
    data = pd.read_csv(file_path, encoding='utf-8')

    # Display the first 10 rows of the dataset
    print("Data Preview:")
    print(data.head(-1)) 
    return data


def get_date_year(date: str):
    '''
    Gets the year from a date string in the format 'dd/mm/yyyy'.
    '''
    return date.split('/')[-1]  # last element is the year


# Function for creating graph of words (notebook)
def find_keywords(speeches, window_size):
    # Merge all speeches into one list of tokens
    tokens = ' '.join(speeches).split()
    n = len(tokens)
    G: nx.DiGraph = nx.DiGraph()
    for i in range(n):
        for j in range(i+1,i+window_size):
            if j < n and tokens[i] != tokens[j]:
                G.add_edge(tokens[i], tokens[j])
    
    # k-core decomposition to undirected graph
    core: nx.Graph = nx.k_core(G.to_undirected())
    return core.nodes(), G


# Function for printing graph info
def print_graph_info(graph: nx.DiGraph):
    """
    Print the nodes with the highest in-degree and out-degree for a given graph.
    """
    in_degrees = dict(graph.in_degree())
    sorted_in_degrees = sorted(in_degrees.items(), key=lambda item: item[1], reverse=True)
    print('Nodes with highest in-degree: ', sorted_in_degrees[:5])

    out_degrees = dict(graph.out_degree())
    sorted_out_degrees = sorted(out_degrees.items(), key=lambda item: item[1], reverse=True)
    print('Nodes with highest out-degree: ', sorted_out_degrees[:5], '\n')


def find_keywords_by_group(data: pd.DataFrame, group_by: str):
    # Creation of dictionary of the groups (members or parties) and their respective speeches
    group_speeches = {}
    for _, row in data.iterrows():
        group_name = str(row[group_by])
        speech = str(row['speech'])  # is preprocessed already!!
        year = get_date_year(str(row['sitting_date']))
        
        # If the group already exists in the dictionary, append the speech
        if group_name not in group_speeches:
            # Create a new entry for the group in the specific year
            group_speeches[group_name] = {year: [speech]}
        elif group_name in group_speeches and year in group_speeches[group_name]:
            group_speeches[group_name][year].append(speech)
        else:
            # Group exists but the year is not in the dictionary
            group_speeches[group_name][year] = [speech]
           
           
    # k-core decomposition for each group's speeches for each year
    group_keywords_yearly = {}

    for group_name, years in group_speeches.items():
        group_keywords_yearly[group_name] = {}
        for year, speeches in years.items():
            # Create a word co-occurrence graph for this group in this year
            try:
                keywords, _ = find_keywords(speeches, 3)
            except Exception as e:
                # Πετάει κάτι error για max() iterable argument is empty αλλα τα βρίσκει κανονικά τα keywords
                continue

            # Store the string keywords in the dictionary
            if keywords:
                group_keywords_yearly[group_name][year] = [str(keyword) for keyword in keywords]
            else:
                group_keywords_yearly[group_name][year] = []

    return group_keywords_yearly
    

def main():
    # Load the processed data
    data = load_file()

    # ---------------------TESTING---------------------
    test = False
    if test:
        # Preprocess the 'speech' column for all speeches
        preprocessed_speeches = data['speech']
        
        # k-core decomposition for all speeches
        keywords, G_corpus = find_keywords(preprocessed_speeches, 3)

        print(G_corpus.number_of_nodes())
        print(G_corpus.number_of_edges())

        print_graph_info(G_corpus)

        core = nx.k_core(G_corpus)
        k_value = max(nx.core_number(G_corpus).values())

        print(f"The k-value of the k-core is: {k_value}")
        print(f"Nodes in the k-core: {list(core.nodes())}")
    
    # -------------------RESULTS-----------------------

    print('Find keywords for each parliament member')
    member_keywords = find_keywords_by_group(data, 'member_name')

    print('Find keywords for each political party')
    party_keywords = find_keywords_by_group(data, 'political_party')

    
    # Save the results in two different json files
    with open("member_keywords.json", "w", encoding='utf-8') as f:
        json.dump(member_keywords, f, ensure_ascii=False, indent=4)

    with open("party_keywords.json", "w", encoding='utf-8') as f:
        json.dump(party_keywords, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()

