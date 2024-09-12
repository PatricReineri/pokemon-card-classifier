import os
import requests
from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import RestClient
import json

RestClient.configure('12345678-1234-1234-1234-123456789ABC')

def download_pokemon_images_and_data(set_id):
    try:
        cards = Card.where(q=f'set.id:{set_id}')
        
        if cards:
            print(f"Scaricate {len(cards)} carte dal set {set_id}.")
            save_images_and_data_to_directory(cards, set_id)
        else:
            print(f"Nessuna carta trovata per il set {set_id}.")
    
    except Exception as err:
        print(f"An error occurred: {err}")

def serialize_card(card):
    card_data = {}
    for key, value in card.__dict__.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            card_data[key] = value
        elif isinstance(value, list):
            card_data[key] = [serialize_card(v) if hasattr(v, '__dict__') else v for v in value]
        elif hasattr(value, '__dict__'):
            card_data[key] = serialize_card(value)
        else:
            card_data[key] = str(value)
    return card_data

def save_images_and_data_to_directory(cards, set_id):
    directory = "base1"
    json_directory = directory
    images_directory = f"{directory}_images"
    os.makedirs(json_directory, exist_ok=True)
    os.makedirs(images_directory, exist_ok=True)
    
    for card in cards:
        card_id = card.id
        image_url = card.images.large
        image_filename = os.path.join(images_directory, f"{card_id}.png")
        json_filename = os.path.join(json_directory, f"{card_id}.json")
        
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            with open(image_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"Immagine salvata per {card_id} in {image_filename}.")

            card_data = serialize_card(card)
            with open(json_filename, 'w') as json_file:
                json.dump(card_data, json_file, indent=4)
            
            print(f"Dati JSON salvati per {card_id} in {json_filename}.")
        
        except requests.exceptions.RequestException as e:
            print(f"Errore durante il download dell'immagine per la carta {card_id}: {e}")
        except Exception as e:
            print(f"Errore durante il salvataggio dei dati JSON per la carta {card_id}: {e}")

print(Set.all())

for set_info in Set.all():
    set_id = set_info.id
    print(f"Downloading images and data for set: {set_info.ptcgoCode}")
    if set_info.name in ["Paldea Evolved"]:
        download_pokemon_images_and_data(set_id)
