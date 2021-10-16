import models
import schemas
from sqlalchemy.orm import Session


def get_otp_user(db: Session, otp: str, id: int):
    return db.query(models.AuthUser).filter(models.AuthUser.otp == otp, models.AuthUser.id == id).first()


def get_user_mobile(db: Session, mobile: str):
    return db.query(models.AuthUser).filter(models.AuthUser.mobile == mobile).first()


def create_user(db: Session, user: schemas.UserInfoBase):
    otp = "123456"
    db_user = models.AuthUser(username=user.username, otp=otp, mobile=user.mobile)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
