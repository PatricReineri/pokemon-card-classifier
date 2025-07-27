import os
import json
import requests
import asyncio
from tcgdexsdk import TCGdex
from tcgdexsdk.enums import Quality, Extension

async def serialize_card(card, tcg):
    # Fetch full card details
    card_details = await tcg.card.get(card.id)
    
    # Fetch full set details if necessary
    set_details = await tcg.set.get(card_details.set.id) if hasattr(card_details, "set") else None
    
    return {
        "id": card.id,
        "name": card.name,
        "image": card.get_image_url(Quality.HIGH, Extension.PNG),
        "rarity": getattr(card, "rarity", None),  # Use getattr to handle missing attributes
        "set": set_details.id if set_details else None,  # Fetch set details if available
        "series": set_details.series.id if set_details and hasattr(set_details, "series") else None,  # Fetch series details if available
        # Add other relevant attributes as needed
    }

async def save_images_and_data_to_directory(cards, set_id, tcg):
    base_dir = f"base1"
    json_directory = base_dir
    images_directory = f"{base_dir}_dimages"

    os.makedirs(json_directory, exist_ok=True)
    os.makedirs(images_directory, exist_ok=True)

    for card in cards:
        card_id = card.id  # Use dot notation to access attributes
        image_url = card.get_image_url(Quality.HIGH, Extension.PNG)  # Use get_image_url with enums

        if not image_url:
            print(f"Nessuna immagine trovata per {card_id}.")
            continue

        image_filename = os.path.join(images_directory, f"{card_id}.png")
        json_filename = os.path.join(json_directory, f"{card_id}.json")

        try:
            response = requests.get(image_url)
            response.raise_for_status()

            with open(image_filename, 'wb') as f:
                f.write(response.content)

            print(f"Immagine salvata per {card_id} in {image_filename}.")

            card_data = await serialize_card(card, tcg)  # Pass the SDK instance to fetch details
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(card_data, f, indent=4, ensure_ascii=False)

            print(f"Dati JSON salvati per {card_id} in {json_filename}.")

        except Exception as e:
            print(f"Errore per la carta {card_id}: {e}")

async def download_pokemon_images_and_data(set_id):
    try:
        tcg = TCGdex("en")
        set_obj = await tcg.set.get(set_id)  # Use async method to fetch set details
        cards = set_obj.cards

        if cards:
            print(f"Scaricate {len(cards)} carte dal set {set_obj.name} ({set_obj.cardCount.total} carte).")
            await save_images_and_data_to_directory(cards, set_id, tcg)  # Pass the SDK instance
        else:
            print(f"Nessuna carta trovata per il set {set_id}.")

    except Exception as err:
        print(f"Errore durante il recupero del set {set_id}: {err}")

async def main():
    tcg = TCGdex("en")
    all_sets = await tcg.serie.get("Scarlet & Violet")  # Fetch the series "Scarlet & Violet"
    print(f"Serie: {all_sets.name} ({len(all_sets.sets)} set)")

    for set_info in all_sets.sets:
        print(f"Trovato set: {set_info.name} ({set_info.id})")
        if set_info.name == "Paldea Evolved":
            await download_pokemon_images_and_data(set_info.id)

# Run the main function
asyncio.run(main())
