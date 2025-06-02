from apscheduler.schedulers.background import BackgroundScheduler
from utils import generate_affirmation

def send_daily_email():
    affirmation = generate_affirmation()
    print("Pretending to send email with affirmation:", affirmation)

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_daily_email, 'cron', hour=8)
    scheduler.start()
