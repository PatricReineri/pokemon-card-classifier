import os
import json
import requests
import asyncio
from tcgdexsdk import TCGdex
from tcgdexsdk.enums import Quality, Extension

async def save_images_and_data_to_directory(cards):
    base_dir = f"card_images"

    os.makedirs(base_dir, exist_ok=True)

    for card in cards:
        card_id = card.id  
        image_url = card.get_image_url(Quality.HIGH, Extension.PNG)  

        if not image_url:
            print(f"Nessuna immagine trovata per {card_id}.")
            continue

        image_filename = os.path.join(base_dir, f"{card_id}.png")

        try:
            response = requests.get(image_url)
            response.raise_for_status()

            with open(image_filename, 'wb') as f:
                f.write(response.content)

            print(f"Immagine salvata per {card_id} in {image_filename}.")

        except Exception as e:
            print(f"Errore per la carta {card_id}: {e}")

async def download_pokemon_images_and_data(set_id):
    try:
        tcg = TCGdex("en")
        set_obj = await tcg.set.get(set_id)  # Use async method to fetch set details
        cards = set_obj.cards

        if cards:
            print(f"Scaricate {len(cards)} carte dal set {set_obj.name} ({set_obj.cardCount.total} carte).")
            await save_images_and_data_to_directory(cards)  # Pass the SDK instance
        else:
            print(f"Nessuna carta trovata per il set {set_id}.")

    except Exception as err:
        print(f"Errore durante il recupero del set {set_id}: {err}")

async def download_all_sets():
    tcg = TCGdex("en")
    all_sets = await tcg.serie.get("Scarlet & Violet")  # Fetch the series "Scarlet & Violet"
    print(f"Serie: {all_sets.name} ({len(all_sets.sets)} set)")

    for set_info in all_sets.sets:
        print(f"Trovato set: {set_info.name} ({set_info.id})")
        await download_pokemon_images_and_data(set_info.id)

if __name__ == "__main__":
    asyncio.run(download_all_sets())
