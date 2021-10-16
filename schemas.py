from pydantic import BaseModel


class UserInfoBase(BaseModel):
    username: str
    mobile: str


class UserInfo(UserInfoBase):
    id: int

    class Config:
        orm_mode = True


class OtpInfo(BaseModel):
    id: int
    otp: str
