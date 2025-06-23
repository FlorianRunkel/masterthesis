from flask import Blueprint, request, jsonify, current_app
from linkedin_api import Linkedin
from backend.config import Config
import logging
import json
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import requests

# Blueprint für LinkedIn-Routen
linkedin_bp = Blueprint('linkedin_bp', __name__)

# Globale Variable für die API-Instanz
linkedin_api = None
selenium_driver = None

def get_linkedin_api_instance():
    """Erstellt oder gibt die bestehende LinkedIn API-Instanz zurück."""
    global linkedin_api
    if linkedin_api is None:
        try:
            email = Config.LINKEDIN_EMAIL
            password = Config.LINKEDIN_PASSWORD
            if not email or not password:
                logging.error("LinkedIn-Anmeldeinformationen sind nicht in der Konfiguration gesetzt.")
                return None
            linkedin_api = Linkedin(email, password, refresh_cookies=True)
            logging.info("LinkedIn API erfolgreich initialisiert")
        except Exception as e:
            logging.error(f"Fehler bei der LinkedIn API Initialisierung: {str(e)}")
            linkedin_api = None # Bei Fehler zurücksetzen
            return None
    return linkedin_api

def get_selenium_driver():
    """Erstellt oder gibt den Selenium WebDriver zurück."""
    global selenium_driver
    if selenium_driver is None:
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Headless-Modus
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Verwende den lokalen Chrome Driver falls vorhanden, sonst automatisch herunterladen
            try:
                selenium_driver = webdriver.Chrome(options=chrome_options)
            except:
                from webdriver_manager.chrome import ChromeDriverManager
                selenium_driver = webdriver.Chrome(
                    ChromeDriverManager().install(),
                    options=chrome_options
                )
            logging.info("Selenium WebDriver erfolgreich initialisiert")
        except Exception as e:
            logging.error(f"Fehler bei der Selenium WebDriver Initialisierung: {str(e)}")
            selenium_driver = None
            return None
    return selenium_driver

def scrape_linkedin_with_selenium(linkedin_url):
    """Scrapt LinkedIn-Profil mit Selenium als Fallback."""
    driver = get_selenium_driver()
    if not driver:
        return None
    
    try:
        logging.info(f"Starte Selenium-Scraping für: {linkedin_url}")
        driver.get(linkedin_url)
        time.sleep(3)  # Warte bis Seite geladen ist
        
        # Warte auf wichtige Elemente
        wait = WebDriverWait(driver, 10)
        
        # Extrahiere Profildaten
        profile_data = {
            'name': '',
            'currentTitle': '',
            'location': '',
            'imageUrl': '',
            'experience': [],
            'education': [],
            'industry': '',
            'summary': ''
        }
        
        try:
            # Name
            name_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.text-heading-xlarge")))
            profile_data['name'] = name_element.text.strip()
        except TimeoutException:
            logging.warning("Name konnte nicht extrahiert werden")
        
        try:
            # Titel
            title_element = driver.find_element(By.CSS_SELECTOR, ".text-body-medium.break-words")
            profile_data['currentTitle'] = title_element.text.strip()
        except NoSuchElementException:
            logging.warning("Titel konnte nicht extrahiert werden")
        
        try:
            # Standort
            location_element = driver.find_element(By.CSS_SELECTOR, ".text-body-small.inline.t-black--light.break-words")
            profile_data['location'] = location_element.text.strip()
        except NoSuchElementException:
            logging.warning("Standort konnte nicht extrahiert werden")
        
        # Erfahrungen
        try:
            experience_section = driver.find_element(By.ID, "experience")
            experience_items = experience_section.find_elements(By.CSS_SELECTOR, ".pvs-list__item--line-separated")
            
            for item in experience_items[:5]:  # Maximal 5 Erfahrungen
                try:
                    title_element = item.find_element(By.CSS_SELECTOR, ".t-bold span[aria-hidden='true']")
                    company_element = item.find_element(By.CSS_SELECTOR, ".t-normal span[aria-hidden='true']")
                    
                    experience = {
                        'title': title_element.text.strip(),
                        'company': company_element.text.strip(),
                        'startDate': '',
                        'endDate': 'Present'
                    }
                    profile_data['experience'].append(experience)
                except NoSuchElementException:
                    continue
        except NoSuchElementException:
            logging.warning("Erfahrungen konnten nicht extrahiert werden")
        
        # Ausbildung
        try:
            education_section = driver.find_element(By.ID, "education")
            education_items = education_section.find_elements(By.CSS_SELECTOR, ".pvs-list__item--line-separated")
            
            for item in education_items[:3]:  # Maximal 3 Ausbildungen
                try:
                    school_element = item.find_element(By.CSS_SELECTOR, ".t-bold span[aria-hidden='true']")
                    degree_element = item.find_element(By.CSS_SELECTOR, ".t-normal span[aria-hidden='true']")
                    
                    education = {
                        'degree': degree_element.text.strip(),
                        'school': school_element.text.strip(),
                        'startDate': '',
                        'endDate': ''
                    }
                    profile_data['education'].append(education)
                except NoSuchElementException:
                    continue
        except NoSuchElementException:
            logging.warning("Ausbildung konnte nicht extrahiert werden")
        
        logging.info(f"Selenium-Scraping erfolgreich für: {profile_data['name']}")
        return profile_data
        
    except Exception as e:
        logging.error(f"Fehler beim Selenium-Scraping: {str(e)}")
        return None

def scrape_linkedin_simple(linkedin_url):
    """Einfaches LinkedIn-Scraping mit requests und BeautifulSoup."""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        logging.info(f"Starte einfaches Scraping für: {linkedin_url}")
        
        # Füge zufällige Verzögerung hinzu
        time.sleep(2)
        
        response = requests.get(linkedin_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        profile_data = {
            'name': '',
            'currentTitle': '',
            'location': '',
            'imageUrl': '',
            'experience': [],
            'education': [],
            'industry': '',
            'summary': ''
        }
        
        # Name extrahieren
        try:
            name_element = soup.find('h1', class_='text-heading-xlarge')
            if name_element:
                profile_data['name'] = name_element.get_text(strip=True)
        except Exception as e:
            logging.warning(f"Name konnte nicht extrahiert werden: {e}")
        
        # Titel extrahieren
        try:
            title_element = soup.find('div', class_='text-body-medium break-words')
            if title_element:
                profile_data['currentTitle'] = title_element.get_text(strip=True)
        except Exception as e:
            logging.warning(f"Titel konnte nicht extrahiert werden: {e}")
        
        # Standort extrahieren
        try:
            location_element = soup.find('span', class_='text-body-small inline t-black--light break-words')
            if location_element:
                profile_data['location'] = location_element.get_text(strip=True)
        except Exception as e:
            logging.warning(f"Standort konnte nicht extrahiert werden: {e}")
        
        # Erfahrungen extrahieren (vereinfacht)
        try:
            experience_section = soup.find('section', {'id': 'experience'})
            if experience_section:
                experience_items = experience_section.find_all('li', class_='pvs-list__item--line-separated')
                
                for item in experience_items[:5]:  # Maximal 5 Erfahrungen
                    try:
                        title_element = item.find('span', class_='t-bold')
                        company_element = item.find('span', class_='t-normal')
                        
                        if title_element and company_element:
                            experience = {
                                'title': title_element.get_text(strip=True),
                                'company': company_element.get_text(strip=True),
                                'startDate': '',
                                'endDate': 'Present'
                            }
                            profile_data['experience'].append(experience)
                    except Exception as e:
                        continue
        except Exception as e:
            logging.warning(f"Erfahrungen konnten nicht extrahiert werden: {e}")
        
        # Ausbildung extrahieren (vereinfacht)
        try:
            education_section = soup.find('section', {'id': 'education'})
            if education_section:
                education_items = education_section.find_all('li', class_='pvs-list__item--line-separated')
                
                for item in education_items[:3]:  # Maximal 3 Ausbildungen
                    try:
                        school_element = item.find('span', class_='t-bold')
                        degree_element = item.find('span', class_='t-normal')
                        
                        if school_element and degree_element:
                            education = {
                                'degree': degree_element.get_text(strip=True),
                                'school': school_element.get_text(strip=True),
                                'startDate': '',
                                'endDate': ''
                            }
                            profile_data['education'].append(education)
                    except Exception as e:
                        continue
        except Exception as e:
            logging.warning(f"Ausbildung konnte nicht extrahiert werden: {e}")
        
        logging.info(f"Einfaches Scraping erfolgreich für: {profile_data['name']}")
        return profile_data
        
    except Exception as e:
        logging.error(f"Fehler beim einfachen Scraping: {str(e)}")
        return None

@linkedin_bp.route('/scrape-linkedin', methods=['POST'])
def scrape_linkedin():
    """Scrapt ein LinkedIn-Profil und gibt die formatierten Daten zurück."""
    
    try:
        data = request.get_json()
        linkedin_url = data.get('url')

        if not linkedin_url:
            return jsonify({'error': 'Keine LinkedIn-URL angegeben'}), 400

        try:
            username = linkedin_url.split('/in/')[1].split('/')[0]
            username = username.split('?')[0]
        except IndexError:
            return jsonify({'error': f"Ungültiges LinkedIn-URL-Format: {linkedin_url}"}), 400

        logging.info(f"Starte LinkedIn-Scraping für: {username}")

        # 1. Versuche zuerst die linkedin-api
        api = get_linkedin_api_instance()
        if api:
            try:
                profile = api.get_profile(username)
                if profile:
                    # Profildaten formatieren (wie bisher)
                    profile_data = {
                        'name': f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
                        'currentTitle': profile.get('headline', ''),
                        'location': profile.get('locationName', ''),
                        'imageUrl': profile.get('displayPictureUrl', '') + profile.get('img_400_400', ''),
                        'experience': [],
                        'education': []
                    }

                    for position in profile.get('experience', []):
                        start = position.get('timePeriod', {}).get('startDate', {})
                        start_year = str(start.get('year', ''))
                        start_month = str(start.get('month', '')).zfill(2) if start.get('month') else '01'
                        start_day = str(start.get('day', '')).zfill(2) if start.get('day') else '01'
                        start_date = f"{start_day}/{start_month}/{start_year}" if start_year else ''
                        end = position.get('timePeriod', {}).get('endDate', {})
                        end_date = 'Present'
                        if end:
                            end_year = str(end.get('year', ''))
                            end_month = str(end.get('month', '')).zfill(2) if end.get('month') else '01'
                            end_day = str(end.get('day', '')).zfill(2) if end.get('day') else '01'
                            end_date = f"{end_day}/{end_month}/{end_year}" if end_year else 'Present'
                        
                        profile_data['experience'].append({
                            'title': position.get('title', ''),
                            'company': position.get('companyName', ''),
                            'startDate': start_date,
                            'endDate': end_date
                        })

                    profile_data['industry'] = profile.get('industry', '')
                    profile_data['summary'] = profile.get('summary', '')

                    for edu in profile.get('education', []):
                        profile_data['education'].append({
                            'degree': edu.get('degreeName', ''),
                            'school': edu.get('schoolName', ''),
                            'startDate': str(edu.get('timePeriod', {}).get('startDate', {}).get('year', '')) if edu.get('timePeriod', {}).get('startDate') else '',
                            'endDate': str(edu.get('timePeriod', {}).get('endDate', {}).get('year', '')) if edu.get('timePeriod', {}).get('endDate') else ''
                        })

                    logging.info(f"LinkedIn API erfolgreich für: {profile_data['name']}")
                    return jsonify(profile_data)

            except Exception as api_error:
                logging.warning(f"LinkedIn API fehlgeschlagen: {str(api_error)}")
                # Fallback zu Selenium
                pass

        # 2. Fallback: Selenium-Scraping
        logging.info("Versuche Selenium-Scraping als Fallback")
        profile_data = scrape_linkedin_with_selenium(linkedin_url)
        
        if profile_data:
            return jsonify(profile_data)
        
        # 3. Fallback: Einfaches Scraping
        logging.info("Versuche einfaches Scraping als letzter Fallback")
        profile_data = scrape_linkedin_simple(linkedin_url)
        
        if profile_data:
            return jsonify(profile_data)
        else:
            return jsonify({
                'error': 'LinkedIn-Profil konnte nicht abgerufen werden.',
                'message': 'Alle Scraping-Methoden sind fehlgeschlagen.',
                'suggestion': 'Versuchen Sie es später erneut oder geben Sie die Daten manuell ein.'
            }), 500

    except Exception as e:
        logging.error(f"Unerwarteter Fehler beim LinkedIn-Scraping: {str(e)}")
        return jsonify({'error': str(e)}), 500 