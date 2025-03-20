# scraping.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging
import json
import pickle
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class HorseRaceDataCollector:
    def __init__(self):
        self.base_url = "https://heppa.hippos.fi/heppa2_backend/"
        self.headers = {'Content-Type': 'application/json'}
        self.cache = {}
        self.load_cache()

    def _make_request(self, endpoint):
        if endpoint in self.cache:
            logger.debug(f"Using cached data for {endpoint}")
            return self.cache[endpoint]
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Making request to {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.cache[endpoint] = data
            logger.debug(f"Response from {url}: {data[:2] if isinstance(data, list) else data}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def load_cache(self):
        dir = './horse-race-predictor/racedata/'
        file = 'api_cache.pkl'
        cache_path = dir + file
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)
                logger.info("Loaded existing API cache")

    def save_cache(self):
        dir = './horse-race-predictor/racedata/'
        file = 'api_cache.pkl'
        cache_path = dir + file        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
            logger.info("Saved API cache")

    def get_races(self, start_date, end_date):
        endpoint = f"race/search/{start_date}/{end_date}/"
        data = self._make_request(endpoint)
        logger.debug(f"Got {len(data) if data else 0} races for {start_date} to {end_date}")
        return data or []

    def get_race_details(self, date, track_code):
        endpoint = f"race/{date}/{track_code}/races"
        data = self._make_request(endpoint)
        logger.debug(f"Got {len(data) if data else 0} race details for {date}, {track_code}")
        return data or []

    def get_start_horses(self, date, track_code, start_number):
        endpoint = f"race/{date}/{track_code}/start/{start_number}"
        data = self._make_request(endpoint)
        logger.debug(f"Got {len(data) if data else 0} starts for {date}, {track_code}, {start_number}")
        return data or []

    def get_horse_stats(self, horse_id):
        if not horse_id:
            logger.debug("No horse_id provided, returning empty dict")
            return {}
        endpoint = f"horse/{horse_id}/stats"
        return self._make_request(endpoint) or {}

    def get_driver_stats(self, driver_id):
        if not driver_id:
            logger.debug("No driver_id provided, returning empty dict")
            return {}
        endpoint = f"driver/{driver_id}/stats"
        return self._make_request(endpoint) or {}

    def collect_data(self, start_date, end_date):
        all_race_data = {}
        races = self.get_races(start_date, end_date)
        if not races:
            logger.warning(f"No races found for {start_date} to {end_date}")
            return all_race_data

        for race in tqdm(races, desc=f"Processing races {start_date}"):
            race_date = race.get('raceDate')
            logger.debug(f"Processing race date: {race_date}")
            all_race_data[race_date] = {}
            for event in race.get('events', []):
                track_code = event.get('trackCode')
                
                if not race_date or not track_code:
                    logger.warning(f"Skipping event due to missing info: {event}")
                    continue
                    
                race_details = self.get_race_details(race_date, track_code)
                if not race_details:
                    logger.warning(f"No race details for {race_date}, {track_code}")
                    continue
                
                all_race_data[race_date][track_code] = {}
                for race_entry in tqdm(race_details, desc=f"Races for {race_date}, {track_code}", leave=False):
                    start_number = race_entry.get('race', {}).get('startNumber')
                    if not start_number:
                        logger.debug(f"No startNumber in race_entry: {race_entry}")
                        continue
                    
                    starts = self.get_start_horses(race_date, track_code, start_number)
                    if not starts:
                        logger.warning(f"No starts for {race_date}, {track_code}, {start_number}")
                        continue
                    
                    # Collect stats for each horse and driver
                    enriched_starts = []
                    for horse in starts:
                        horse_id = horse.get('horseId', '')
                        driver_id = horse.get('driverId', '')
                        horse_stats = self.get_horse_stats(horse_id)
                        driver_stats = self.get_driver_stats(driver_id)
                        enriched_starts.append({
                            'horse_data': horse,
                            'horse_stats': horse_stats,
                            'driver_stats': driver_stats
                        })
                    
                    all_race_data[race_date][track_code][start_number] = {
                        'race_info': race_entry,
                        'starts': enriched_starts
                    }
        
        return all_race_data

def scrape_periods(start_year, end_year, chunk_size='month'):
    data_collector = HorseRaceDataCollector()
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    current_date = start_date
    data_files = []
    data_dir = './horse-race-predictor/racedata/'

    logger.info(f"Starting scrape from {start_year} to {end_year}")
    while current_date < end_date:
        if chunk_size == 'month':
            next_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        elif chunk_size == 'year':
            next_date = current_date.replace(year=current_date.year + 1, month=12, day=31)
        next_date = min(next_date, end_date)
        
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = next_date.strftime('%Y-%m-%d')
        filename = os.path.join(data_dir, f"scraped_race_data_{start_str}_{end_str}.json")
        
        if not os.path.exists(filename):
            logger.info(f"Scraping {start_str} to {end_str}")
            race_data = data_collector.collect_data(start_str, end_str)
            if race_data:
                with open(filename, 'w') as f:
                    json.dump(race_data, f, indent=2)
                logger.info(f"Saved data to {filename}")
            else:
                logger.warning(f"No data collected for {start_str} to {end_str}")
            data_collector.save_cache()
        else:
            logger.info(f"Skipping {start_str} to {end_str} - {filename} already exists")
        
        data_files.append(filename)
        current_date = next_date + timedelta(days=1)
    
    return data_files

if __name__ == "__main__":
    # Scrape a test range (e.g., 2024)
    data_files = scrape_periods(2010, 2016, chunk_size='month')
    logger.info(f"Scraped data saved to: {data_files}")
