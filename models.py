from sqlalchemy import Column, Integer, String
from database import Base


class AuthUser(Base):
    __tablename__ = "auth_user"

    id = Column(Integer, primary_key=True, index=True)
    mobile = Column(String)
    username = Column(String)
    otp = Column(String)
