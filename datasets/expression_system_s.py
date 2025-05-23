import pandas as pd
import requests

def get_expression_system_graphql(pdb_id):
    url = "https://data.rcsb.org/graphql"
    query = """
    {
      entry(entry_id: "%s") {
        polymer_entities {
          rcsb_entity_host_organism {
            expression_system
          }
        }
      }
    }
    """ % pdb_id

    response = requests.post(url, json={'query': query})
    if response.status_code == 200:
        data = response.json()
        if data is not None and 'data' in data and data['data'] is not None and \
           'entry' in data['data'] and data['data']['entry'] is not None and \
           'polymer_entities' in data['data']['entry']:
            entities = data['data']['entry']['polymer_entities']
            expression_systems = [entity['rcsb_entity_host_organism'][0]['expression_system']
                                  for entity in entities
                                  if entity['rcsb_entity_host_organism'] and entity['rcsb_entity_host_organism'][0]['expression_system']]
            return ', '.join(list(set(expression_systems))) if expression_systems else ""
        else:
            return ""
    else:
        return ""

def update_csv_with_expression_system(csv_file_path, output_file_path):
    df = pd.read_csv(csv_file_path)
    df['Expression System'] = df['pdb'].apply(get_expression_system_graphql)
    print(df)
    df.to_csv(output_file_path, index=False)

# Specify the path to your CSV file and where to save the updated file
input_csv_path = ''  # Replace with your actual input CSV file path
output_csv_path = ''  # Replace with your desired output CSV file path

update_csv_with_expression_system(input_csv_path, output_csv_path)
print("Finish")
