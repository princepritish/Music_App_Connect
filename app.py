import models
import schemas
import crud
from sqlalchemy.orm import Session
from fastapi import Depends, FastAPI, HTTPException
from database import engine, SessionLocal
from ML_functions import *


app = FastAPI()

models.Base.metadata.create_all(bind=engine)


def get_db():
    db = None
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.post("/user", response_model=schemas.UserInfo)
def create_user(user: schemas.UserInfoBase, db: Session = Depends(get_db)):
    dbuser = crud.get_user_mobile(db, mobile=user.mobile)
    if dbuser:
        raise HTTPException(status_code=400, detail="Mobile number already exists")
    return crud.create_user(db=db, user=user)


@app.post("/user/otp")
def verify_otp(user: schemas.OtpInfo, db: Session = Depends(get_db)):
    user_otp = crud.get_otp_user(db, otp=user.otp, id=user.id)
    if not user_otp:
        raise HTTPException(status_code=400, detail="Invalid otp")
    return {"status": "success", "message": "OTP verified"}


@app.get("/")
def root():
    return {"Music App": "Welcome to Music App"}

@app.post("/user/get_mood")
def get_mood(filepath):
    data = prepare_data(audio_link, n=n_mfcc)
    mean, std = load_mean_std()
    data = (data - mean) / std
    print(data.shape)
    model = tf.keras.models.load_model('ml_model/my_model')
    probs = model.predict(np.expand_dims(data, axis=0))
    print(f"Prediction class is {class_mapping[np.argmax(probs)]}")
    return {
        "Class " : class_mapping[np.argmax(probs)]
    }