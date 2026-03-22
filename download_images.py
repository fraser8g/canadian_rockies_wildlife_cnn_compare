'''
iNaturalist API - "GET /observations" was used for 
downloading wildlife images for particular species by taxon_id
and a custom region mapped out on the observations screen (iNaturalist)

Documentation found here:
https://api.inaturalist.org/v1/docs/#!/Observations/get_observations
'''

import os
import requests

# CONFIGS
IMAGES_PER_CLASS = 500
OUTPUT_DIR = "inat_dataset"

SPECIES = {
    "bighorn_sheep": 145538, #Rocky Mountain Bighorn Sheep
    "black_bear": 41638, #American Black Bear
    "caribou": 42199, #Caribou
    "cougar": 143589, #North American Mountain Lion
    "coyote": 42051, #Coyote
    "deer": 42219, #White-tailed and Mule Deer
    "elk": 204114, #Wapiti
    "fox": 42069, #Red Fox
    "grizzly_bear": 125461, #Grizzly Bear
    "lynx": 41973, #Lynxes and Bobcat
    "moose": 522193, #Moose
    "mountain_goat": 42413, #Mountain Goat
    "wolf": 42048, #Gray Wolf
    "wolverine": 236285 #American Wolverine
}

API_URL = "https://api.inaturalist.org/v1/observations"

def download_image(url, filepath):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            #write binary for extension .jpg
            with open(filepath, "wb") as f:
                f.write(r.content)
            return True
    except:
        return False


def fetch_images(species_name, taxon_id, limit):

    folder = os.path.join(OUTPUT_DIR, species_name)
    os.makedirs(folder, exist_ok=True)

    downloaded = 0
    page = 1

    while downloaded < limit:
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "per_page": 100,
            "page": page,
            #Custom region of the Canadian Rockies (iNaturalist)
            "nelat": 65.52517614337124, 
            "nelgn": -110.95378674173236,
            "swlat": 43.60173800199343,
            "swlng": -140.48503674173236
        }

        response = requests.get(API_URL, params=params).json()
        results = response["results"]

        if len(results) == 0:
            break

        for obs in results:
            if downloaded >= limit:
                break

            photos = obs.get("photos", [])

            for photo in photos:
                if downloaded >= limit:
                    break

                url = photo["url"].replace("square", "large")

                filename = f"{species_name}_{downloaded}.jpg"
                filepath = os.path.join(folder, filename)

                success = download_image(url, filepath)

                if success:
                    downloaded += 1
                    print(f"{species_name}: {downloaded}/{limit}")

        page += 1


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for species_name, taxon_id in SPECIES.items():
        print(f"Downloading {species_name}")
        fetch_images(species_name, taxon_id, IMAGES_PER_CLASS)

    print("Download complete")


if __name__ == "__main__":
    main()