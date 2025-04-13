from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Enum as SQLAlchemyEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext

# Database setup
DATABASE_URL = "sqlite:///./mentorship_marketplace.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define Enums
class UserRole(str, Enum):
    MENTOR = "mentor"
    MENTEE = "mentee"
    ADMIN = "admin"

class ExpertiseLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class MeetingFormat(str, Enum):
    VIRTUAL = "virtual"
    IN_PERSON = "in_person"
    HYBRID = "hybrid"

class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    role = Column(SQLAlchemyEnum(UserRole))
    bio = Column(Text, nullable=True)
    profile_picture_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    mentor_profile = relationship("MentorProfile", back_populates="user", uselist=False)
    mentee_profile = relationship("MenteeProfile", back_populates="user", uselist=False)
    reviews_given = relationship("Review", foreign_keys="Review.reviewer_id", back_populates="reviewer")
    reviews_received = relationship("Review", foreign_keys="Review.reviewee_id", back_populates="reviewee")

class MentorProfile(Base):
    __tablename__ = "mentor_profiles"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    expertise_areas = Column(String)  # Comma-separated list of areas
    years_experience = Column(Integer)
    hourly_rate = Column(Float)
    availability = Column(String)  # JSON string with availability schedule
    max_mentees = Column(Integer, default=5)
    current_mentees_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="mentor_profile")
    mentorship_sessions = relationship("MentorshipSession", back_populates="mentor")

class MenteeProfile(Base):
    __tablename__ = "mentee_profiles"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    interests = Column(String)  # Comma-separated list of interests
    goals = Column(Text)
    expertise_level = Column(SQLAlchemyEnum(ExpertiseLevel))
    preferred_meeting_format = Column(SQLAlchemyEnum(MeetingFormat))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="mentee_profile")
    mentorship_sessions = relationship("MentorshipSession", back_populates="mentee")

class MentorshipSession(Base):
    __tablename__ = "mentorship_sessions"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    mentor_id = Column(String, ForeignKey("mentor_profiles.id"))
    mentee_id = Column(String, ForeignKey("mentee_profiles.id"))
    status = Column(SQLAlchemyEnum(Status), default=Status.PENDING)
    start_date = Column(DateTime)
    end_date = Column(DateTime, nullable=True)
    meeting_format = Column(SQLAlchemyEnum(MeetingFormat))
    goals = Column(Text)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    mentor = relationship("MentorProfile", back_populates="mentorship_sessions")
    mentee = relationship("MenteeProfile", back_populates="mentorship_sessions")
    meetings = relationship("Meeting", back_populates="session")

class Meeting(Base):
    __tablename__ = "meetings"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("mentorship_sessions.id"))
    scheduled_time = Column(DateTime)
    duration_minutes = Column(Integer)
    meeting_link = Column(String, nullable=True)
    location = Column(String, nullable=True)
    status = Column(SQLAlchemyEnum(Status), default=Status.PENDING)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("MentorshipSession", back_populates="meetings")

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    reviewer_id = Column(String, ForeignKey("users.id"))
    reviewee_id = Column(String, ForeignKey("users.id"))
    rating = Column(Integer)  # 1-5 stars
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reviewer = relationship("User", foreign_keys=[reviewer_id], back_populates="reviews_given")
    reviewee = relationship("User", foreign_keys=[reviewee_id], back_populates="reviews_received")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models for API
class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    role: UserRole
    bio: Optional[str] = None
    profile_picture_url: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class MentorProfileBase(BaseModel):
    expertise_areas: str
    years_experience: int
    hourly_rate: float
    availability: str
    max_mentees: int = 5

class MentorProfileCreate(MentorProfileBase):
    user_id: str

class MentorProfileResponse(MentorProfileBase):
    id: str
    user_id: str
    current_mentees_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class MentorWithUserResponse(MentorProfileResponse):
    user: UserResponse
    
    class Config:
        orm_mode = True

class MenteeProfileBase(BaseModel):
    interests: str
    goals: str
    expertise_level: ExpertiseLevel
    preferred_meeting_format: MeetingFormat

class MenteeProfileCreate(MenteeProfileBase):
    user_id: str

class MenteeProfileResponse(MenteeProfileBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class MenteeWithUserResponse(MenteeProfileResponse):
    user: UserResponse
    
    class Config:
        orm_mode = True

class MentorshipSessionBase(BaseModel):
    mentor_id: str
    mentee_id: str
    start_date: datetime
    end_date: Optional[datetime] = None
    meeting_format: MeetingFormat
    goals: str
    notes: Optional[str] = None

class MentorshipSessionCreate(MentorshipSessionBase):
    pass

class MentorshipSessionResponse(MentorshipSessionBase):
    id: str
    status: Status
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class MeetingBase(BaseModel):
    session_id: str
    scheduled_time: datetime
    duration_minutes: int
    meeting_link: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None

class MeetingCreate(MeetingBase):
    pass

class MeetingResponse(MeetingBase):
    id: str
    status: Status
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class ReviewBase(BaseModel):
    reviewer_id: str
    reviewee_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class ReviewCreate(ReviewBase):
    pass

class ReviewResponse(ReviewBase):
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Initialize FastAPI
app = FastAPI(
    title="Mentor-Mentee Marketplace API",
    description="API for a marketplace connecting mentors and mentees",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint, returns welcome message.
    """
    return {"message": "Welcome to the Mentor-Mentee Marketplace API"}

# User routes
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user.
    """
    # Check if user with email already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        first_name=user.first_name,
        last_name=user.last_name,
        role=user.role,
        bio=user.bio,
        profile_picture_url=user.profile_picture_url
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/", response_model=List[UserResponse], tags=["Users"])
async def get_users(
    skip: int = 0, 
    limit: int = 100, 
    role: Optional[UserRole] = None, 
    db: Session = Depends(get_db)
):
    """
    Get all users with optional filtering by role.
    """
    query = db.query(User)
    if role:
        query = query.filter(User.role == role)
    
    users = query.offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """
    Get a specific user by ID.
    """
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.put("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def update_user(user_id: str, user: UserBase, db: Session = Depends(get_db)):
    """
    Update a user's information.
    """
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update user fields
    for key, value in user.dict().items():
        setattr(db_user, key, value)
    
    db_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_user)
    return db_user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Users"])
async def delete_user(user_id: str, db: Session = Depends(get_db)):
    """
    Delete a user.
    """
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    return None

# Mentor Profile routes
@app.post("/mentor-profiles/", response_model=MentorProfileResponse, status_code=status.HTTP_201_CREATED, tags=["Mentor Profiles"])
async def create_mentor_profile(profile: MentorProfileCreate, db: Session = Depends(get_db)):
    """
    Create a mentor profile for an existing user.
    """
    # Verify user exists and has the right role
    db_user = db.query(User).filter(User.id == profile.user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    if db_user.role != UserRole.MENTOR:
        raise HTTPException(status_code=400, detail="User must have mentor role")
    
    # Check if profile already exists
    existing_profile = db.query(MentorProfile).filter(MentorProfile.user_id == profile.user_id).first()
    if existing_profile:
        raise HTTPException(status_code=400, detail="Mentor profile already exists for this user")
    
    # Create profile
    db_profile = MentorProfile(**profile.dict())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.get("/mentor-profiles/", response_model=List[MentorWithUserResponse], tags=["Mentor Profiles"])
async def get_mentor_profiles(
    skip: int = 0, 
    limit: int = 100, 
    expertise: Optional[str] = None,
    max_rate: Optional[float] = None,
    min_experience: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get all mentor profiles with optional filtering by expertise, rate, and experience.
    """
    query = db.query(MentorProfile)
    
    if expertise:
        query = query.filter(MentorProfile.expertise_areas.contains(expertise))
    
    if max_rate is not None:
        query = query.filter(MentorProfile.hourly_rate <= max_rate)
    
    if min_experience is not None:
        query = query.filter(MentorProfile.years_experience >= min_experience)
    
    profiles = query.offset(skip).limit(limit).all()
    return profiles

@app.get("/mentor-profiles/{profile_id}", response_model=MentorWithUserResponse, tags=["Mentor Profiles"])
async def get_mentor_profile(profile_id: str, db: Session = Depends(get_db)):
    """
    Get a specific mentor profile by ID.
    """
    db_profile = db.query(MentorProfile).filter(MentorProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    return db_profile

@app.put("/mentor-profiles/{profile_id}", response_model=MentorProfileResponse, tags=["Mentor Profiles"])
async def update_mentor_profile(profile_id: str, profile: MentorProfileBase, db: Session = Depends(get_db)):
    """
    Update a mentor profile.
    """
    db_profile = db.query(MentorProfile).filter(MentorProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Update profile fields
    for key, value in profile.dict().items():
        setattr(db_profile, key, value)
    
    db_profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.delete("/mentor-profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Mentor Profiles"])
async def delete_mentor_profile(profile_id: str, db: Session = Depends(get_db)):
    """
    Delete a mentor profile.
    """
    db_profile = db.query(MentorProfile).filter(MentorProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    db.delete(db_profile)
    db.commit()
    return None

# Mentee Profile routes
@app.post("/mentee-profiles/", response_model=MenteeProfileResponse, status_code=status.HTTP_201_CREATED, tags=["Mentee Profiles"])
async def create_mentee_profile(profile: MenteeProfileCreate, db: Session = Depends(get_db)):
    """
    Create a mentee profile for an existing user.
    """
    # Verify user exists and has the right role
    db_user = db.query(User).filter(User.id == profile.user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    if db_user.role != UserRole.MENTEE:
        raise HTTPException(status_code=400, detail="User must have mentee role")
    
    # Check if profile already exists
    existing_profile = db.query(MenteeProfile).filter(MenteeProfile.user_id == profile.user_id).first()
    if existing_profile:
        raise HTTPException(status_code=400, detail="Mentee profile already exists for this user")
    
    # Create profile
    db_profile = MenteeProfile(**profile.dict())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.get("/mentee-profiles/", response_model=List[MenteeWithUserResponse], tags=["Mentee Profiles"])
async def get_mentee_profiles(
    skip: int = 0, 
    limit: int = 100, 
    interests: Optional[str] = None,
    expertise_level: Optional[ExpertiseLevel] = None,
    meeting_format: Optional[MeetingFormat] = None,
    db: Session = Depends(get_db)
):
    """
    Get all mentee profiles with optional filtering by interests, expertise level, and meeting format.
    """
    query = db.query(MenteeProfile)
    
    if interests:
        query = query.filter(MenteeProfile.interests.contains(interests))
    
    if expertise_level:
        query = query.filter(MenteeProfile.expertise_level == expertise_level)
    
    if meeting_format:
        query = query.filter(MenteeProfile.preferred_meeting_format == meeting_format)
    
    profiles = query.offset(skip).limit(limit).all()
    return profiles

@app.get("/mentee-profiles/{profile_id}", response_model=MenteeWithUserResponse, tags=["Mentee Profiles"])
async def get_mentee_profile(profile_id: str, db: Session = Depends(get_db)):
    """
    Get a specific mentee profile by ID.
    """
    db_profile = db.query(MenteeProfile).filter(MenteeProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Mentee profile not found")
    return db_profile

@app.put("/mentee-profiles/{profile_id}", response_model=MenteeProfileResponse, tags=["Mentee Profiles"])
async def update_mentee_profile(profile_id: str, profile: MenteeProfileBase, db: Session = Depends(get_db)):
    """
    Update a mentee profile.
    """
    db_profile = db.query(MenteeProfile).filter(MenteeProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Mentee profile not found")
    
    # Update profile fields
    for key, value in profile.dict().items():
        setattr(db_profile, key, value)
    
    db_profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.delete("/mentee-profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Mentee Profiles"])
async def delete_mentee_profile(profile_id: str, db: Session = Depends(get_db)):
    """
    Delete a mentee profile.
    """
    db_profile = db.query(MenteeProfile).filter(MenteeProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Mentee profile not found")
    
    db.delete(db_profile)
    db.commit()
    return None

# Mentorship Session routes
@app.post("/mentorship-sessions/", response_model=MentorshipSessionResponse, status_code=status.HTTP_201_CREATED, tags=["Mentorship Sessions"])
async def create_mentorship_session(session: MentorshipSessionCreate, db: Session = Depends(get_db)):
    """
    Create a new mentorship session between a mentor and mentee.
    """
    # Verify mentor and mentee exist
    mentor = db.query(MentorProfile).filter(MentorProfile.id == session.mentor_id).first()
    if mentor is None:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    mentee = db.query(MenteeProfile).filter(MenteeProfile.id == session.mentee_id).first()
    if mentee is None:
        raise HTTPException(status_code=404, detail="Mentee profile not found")
    
    # Check if mentor has capacity
    if mentor.current_mentees_count >= mentor.max_mentees:
        raise HTTPException(status_code=400, detail="Mentor has reached maximum mentee capacity")
    
    # Create session
    db_session = MentorshipSession(**session.dict(), status=Status.PENDING)
    db.add(db_session)
    
    # Update mentor's current mentees count
    mentor.current_mentees_count += 1
    
    db.commit()
    db.refresh(db_session)
    return db_session

@app.get("/mentorship-sessions/", response_model=List[MentorshipSessionResponse], tags=["Mentorship Sessions"])
async def get_mentorship_sessions(
    skip: int = 0, 
    limit: int = 100, 
    mentor_id: Optional[str] = None,
    mentee_id: Optional[str] = None,
    status: Optional[Status] = None,
    db: Session = Depends(get_db)
):
    """
    Get all mentorship sessions with optional filtering by mentor, mentee, and status.
    """
    query = db.query(MentorshipSession)
    
    if mentor_id:
        query = query.filter(MentorshipSession.mentor_id == mentor_id)
    
    if mentee_id:
        query = query.filter(MentorshipSession.mentee_id == mentee_id)
    
    if status:
        query = query.filter(MentorshipSession.status == status)
    
    sessions = query.offset(skip).limit(limit).all()
    return sessions

@app.get("/mentorship-sessions/{session_id}", response_model=MentorshipSessionResponse, tags=["Mentorship Sessions"])
async def get_mentorship_session(session_id: str, db: Session = Depends(get_db)):
    """
    Get a specific mentorship session by ID.
    """
    db_session = db.query(MentorshipSession).filter(MentorshipSession.id == session_id).first()
    if db_session is None:
        raise HTTPException(status_code=404, detail="Mentorship session not found")
    return db_session

@app.put("/mentorship-sessions/{session_id}", response_model=MentorshipSessionResponse, tags=["Mentorship Sessions"])
async def update_mentorship_session(
    session_id: str, 
    session_update: MentorshipSessionBase, 
    db: Session = Depends(get_db)
):
    """
    Update a mentorship session.
    """
    db_session = db.query(MentorshipSession).filter(MentorshipSession.id == session_id).first()
    if db_session is None:
        raise HTTPException(status_code=404, detail="Mentorship session not found")
    
    # Update session fields
    for key, value in session_update.dict().items():
        setattr(db_session, key, value)
    
    db_session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_session)
    return db_session

@app.put("/mentorship-sessions/{session_id}/status", response_model=MentorshipSessionResponse, tags=["Mentorship Sessions"])
async def update_session_status(session_id: str, status: Status, db: Session = Depends(get_db)):
    """
    Update the status of a mentorship session.
    """
    db_session = db.query(MentorshipSession).filter(MentorshipSession.id == session_id).first()
    if db_session is None:
        raise HTTPException(status_code=404, detail="Mentorship session not found")
    
    # Update status
    old_status = db_session.status
    db_session.status = status
    db_session.updated_at = datetime.utcnow()
    
    # If session is completed or rejected, decrease mentor's current mentees count
    if (old_status != Status.COMPLETED and status == Status.COMPLETED) or (old_status != Status.REJECTED and status == Status.REJECTED):
        mentor = db.query(MentorProfile).filter(MentorProfile.id == db_session.mentor_id).first()
        if mentor and mentor.current_mentees_count > 0:
            mentor.current_mentees_count -= 1
    
    db.commit()
    db.refresh(db_session)
    return db_session

@app.delete("/mentorship-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Mentorship Sessions"])
async def delete_mentorship_session(session_id: str, db: Session = Depends(get_db)):
    """
    Delete a mentorship session.
    """
    db_session = db.query(MentorshipSession).filter(MentorshipSession.id == session_id).first()
    if db_session is None:
        raise HTTPException(status_code=404, detail="Mentorship session not found")
    
    # Decrease mentor's current mentees count if session was active
    if db_session.status == Status.APPROVED:
        mentor = db.query(MentorProfile).filter(MentorProfile.id == db_session.mentor_id).first()
        if mentor and mentor.current_mentees_count > 0:
            mentor.current_mentees_count -= 1
    
    db.delete(db_session)
    db.commit()
    return None

# Meeting routes
@app.post("/meetings/", response_model=MeetingResponse, status_code=status.HTTP_201_CREATED, tags=["Meetings"])
async def create_meeting(meeting: MeetingCreate, db: Session = Depends(get_db)):
    """
    Create a new meeting within a mentorship session.
    """
    # Verify session exists
    session = db.query(MentorshipSession).filter(MentorshipSession.id == meeting.session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Mentorship session not found")
    
    # Create meeting
    db_meeting = Meeting(**meeting.dict(), status=Status.PENDING)
    db.add(db_meeting)
    db.commit()
    db.refresh(db_meeting)
    return db_meeting

@app.get("/meetings/", response_model=List[MeetingResponse], tags=["Meetings"])
async def get_meetings(
    skip: int = 0, 
    limit: int = 100, 
    session_id: Optional[str] = None,
    status: Optional[Status] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    db: Session = Depends(get_db)):
    """
    Get all meetings with optional filtering by session, status, and date range.
    """
    query = db.query(Meeting)
    
    if session_id:
        query = query.filter(Meeting.session_id == session_id)
    
    if status:
        query = query.filter(Meeting.status == status)
    
    if from_date:
        query = query.filter(Meeting.scheduled_time >= from_date)
    
    if to_date:
        query = query.filter(Meeting.scheduled_time <= to_date)
    
    meetings = query.offset(skip).limit(limit).all()
    return meetings

@app.get("/meetings/{meeting_id}", response_model=MeetingResponse, tags=["Meetings"])
async def get_meeting(meeting_id: str, db: Session = Depends(get_db)):
    """
    Get a specific meeting by ID.
    """
    db_meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if db_meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return db_meeting

@app.put("/meetings/{meeting_id}", response_model=MeetingResponse, tags=["Meetings"])
async def update_meeting(meeting_id: str, meeting: MeetingBase, db: Session = Depends(get_db)):
    """
    Update a meeting.
    """
    db_meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if db_meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    # Update meeting fields
    for key, value in meeting.dict().items():
        setattr(db_meeting, key, value)
    
    db_meeting.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_meeting)
    return db_meeting

@app.put("/meetings/{meeting_id}/status", response_model=MeetingResponse, tags=["Meetings"])
async def update_meeting_status(meeting_id: str, status: Status, db: Session = Depends(get_db)):
    """
    Update the status of a meeting.
    """
    db_meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if db_meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    # Update status
    db_meeting.status = status
    db_meeting.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_meeting)
    return db_meeting

@app.delete("/meetings/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Meetings"])
async def delete_meeting(meeting_id: str, db: Session = Depends(get_db)):
    """
    Delete a meeting.
    """
    db_meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if db_meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    db.delete(db_meeting)
    db.commit()
    return None

# Review routes
@app.post("/reviews/", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED, tags=["Reviews"])
async def create_review(review: ReviewCreate, db: Session = Depends(get_db)):
    """
    Create a new review for a user.
    """
    # Verify both users exist
    reviewer = db.query(User).filter(User.id == review.reviewer_id).first()
    if reviewer is None:
        raise HTTPException(status_code=404, detail="Reviewer not found")
    
    reviewee = db.query(User).filter(User.id == review.reviewee_id).first()
    if reviewee is None:
        raise HTTPException(status_code=404, detail="Reviewee not found")
    
    # Check if review already exists
    existing_review = db.query(Review).filter(
        Review.reviewer_id == review.reviewer_id,
        Review.reviewee_id == review.reviewee_id
    ).first()
    
    if existing_review:
        raise HTTPException(status_code=400, detail="Review already exists")
    
    # Create review
    db_review = Review(**review.dict())
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

@app.get("/reviews/", response_model=List[ReviewResponse], tags=["Reviews"])
async def get_reviews(
    skip: int = 0, 
    limit: int = 100, 
    reviewer_id: Optional[str] = None,
    reviewee_id: Optional[str] = None,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get all reviews with optional filtering by reviewer, reviewee, and rating range.
    """
    query = db.query(Review)
    
    if reviewer_id:
        query = query.filter(Review.reviewer_id == reviewer_id)
    
    if reviewee_id:
        query = query.filter(Review.reviewee_id == reviewee_id)
    
    if min_rating is not None:
        query = query.filter(Review.rating >= min_rating)
    
    if max_rating is not None:
        query = query.filter(Review.rating <= max_rating)
    
    reviews = query.offset(skip).limit(limit).all()
    return reviews

@app.get("/reviews/{review_id}", response_model=ReviewResponse, tags=["Reviews"])
async def get_review(review_id: str, db: Session = Depends(get_db)):
    """
    Get a specific review by ID.
    """
    db_review = db.query(Review).filter(Review.id == review_id).first()
    if db_review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return db_review

@app.put("/reviews/{review_id}", response_model=ReviewResponse, tags=["Reviews"])
async def update_review(review_id: str, review: ReviewBase, db: Session = Depends(get_db)):
    """
    Update a review.
    """
    db_review = db.query(Review).filter(Review.id == review_id).first()
    if db_review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    
    # Update review fields
    for key, value in review.dict().items():
        setattr(db_review, key, value)
    
    db_review.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_review)
    return db_review

@app.delete("/reviews/{review_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Reviews"])
async def delete_review(review_id: str, db: Session = Depends(get_db)):
    """
    Delete a review.
    """
    db_review = db.query(Review).filter(Review.id == review_id).first()
    if db_review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    
    db.delete(db_review)
    db.commit()
    return None

# Search routes
@app.get("/search/mentors/", response_model=List[MentorWithUserResponse], tags=["Search"])
async def search_mentors(
    query: str = Query(None, description="Search query for mentor name, expertise, or bio"),
    expertise_areas: Optional[str] = None,
    min_experience: Optional[int] = None,
    max_rate: Optional[float] = None,
    availability: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Search for mentors based on multiple criteria.
    """
    # Base query joining mentor profiles with users
    base_query = db.query(MentorProfile).join(User, MentorProfile.user_id == User.id)
    
    # Apply filters
    if query:
        base_query = base_query.filter(
            (User.first_name.ilike(f"%{query}%")) |
            (User.last_name.ilike(f"%{query}%")) |
            (User.bio.ilike(f"%{query}%")) |
            (MentorProfile.expertise_areas.ilike(f"%{query}%"))
        )
    
    if expertise_areas:
        areas_list = expertise_areas.split(',')
        for area in areas_list:
            base_query = base_query.filter(MentorProfile.expertise_areas.ilike(f"%{area.strip()}%"))
    
    if min_experience is not None:
        base_query = base_query.filter(MentorProfile.years_experience >= min_experience)
    
    if max_rate is not None:
        base_query = base_query.filter(MentorProfile.hourly_rate <= max_rate)
    
    if availability:
        base_query = base_query.filter(MentorProfile.availability.ilike(f"%{availability}%"))
    
    # Get results
    mentors = base_query.offset(skip).limit(limit).all()
    return mentors

@app.get("/search/mentees/", response_model=List[MenteeWithUserResponse], tags=["Search"])
async def search_mentees(
    query: str = Query(None, description="Search query for mentee name, interests, goals, or bio"),
    interests: Optional[str] = None,
    expertise_level: Optional[ExpertiseLevel] = None,
    meeting_format: Optional[MeetingFormat] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Search for mentees based on multiple criteria.
    """
    # Base query joining mentee profiles with users
    base_query = db.query(MenteeProfile).join(User, MenteeProfile.user_id == User.id)
    
    # Apply filters
    if query:
        base_query = base_query.filter(
            (User.first_name.ilike(f"%{query}%")) |
            (User.last_name.ilike(f"%{query}%")) |
            (User.bio.ilike(f"%{query}%")) |
            (MenteeProfile.interests.ilike(f"%{query}%")) |
            (MenteeProfile.goals.ilike(f"%{query}%"))
        )
    
    if interests:
        interests_list = interests.split(',')
        for interest in interests_list:
            base_query = base_query.filter(MenteeProfile.interests.ilike(f"%{interest.strip()}%"))
    
    if expertise_level:
        base_query = base_query.filter(MenteeProfile.expertise_level == expertise_level)
    
    if meeting_format:
        base_query = base_query.filter(MenteeProfile.preferred_meeting_format == meeting_format)
    
    # Get results
    mentees = base_query.offset(skip).limit(limit).all()
    return mentees

@app.get("/search/sessions/", response_model=List[MentorshipSessionResponse], tags=["Search"])
async def search_sessions(
    mentor_name: Optional[str] = None,
    mentee_name: Optional[str] = None,
    status: Optional[Status] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Search for mentorship sessions based on multiple criteria.
    """
    # Complex query joining sessions with mentor and mentee profiles and users
    base_query = db.query(MentorshipSession).\
        join(MentorProfile, MentorshipSession.mentor_id == MentorProfile.id).\
        join(User, MentorProfile.user_id == User.id, aliased=True).\
        join(MenteeProfile, MentorshipSession.mentee_id == MenteeProfile.id).\
        join(User, MenteeProfile.user_id == User.id, aliased=True)
    
    # Apply filters
    if mentor_name:
        base_query = base_query.filter(
            (User.first_name.ilike(f"%{mentor_name}%")) |
            (User.last_name.ilike(f"%{mentor_name}%"))
        )
    
    if mentee_name:
        base_query = base_query.filter(
            (User.first_name.ilike(f"%{mentee_name}%")) |
            (User.last_name.ilike(f"%{mentee_name}%"))
        )
    
    if status:
        base_query = base_query.filter(MentorshipSession.status == status)
    
    if from_date:
        base_query = base_query.filter(MentorshipSession.start_date >= from_date)
    
    if to_date:
        base_query = base_query.filter(MentorshipSession.start_date <= to_date)
    
    # Get results
    sessions = base_query.offset(skip).limit(limit).all()
    return sessions

# Authentication and Login route (simplified for this example)
@app.post("/login/", response_model=TokenResponse, tags=["Authentication"])
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    User login endpoint. Returns an access token on successful authentication.
    Note: This is a simplified example - in a real application you would use proper JWT tokens.
    """
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real application, you would generate a proper JWT token here
    token = f"sample_token_{user.id}"
    
    return {"access_token": token, "token_type": "bearer"}

# Analytics routes (simplified examples)
@app.get("/analytics/mentor-stats/{mentor_id}", tags=["Analytics"])
async def get_mentor_stats(mentor_id: str, db: Session = Depends(get_db)):
    """
    Get analytics for a specific mentor.
    """
    # Check if mentor exists
    mentor = db.query(MentorProfile).filter(MentorProfile.id == mentor_id).first()
    if mentor is None:
        raise HTTPException(status_code=404, detail="Mentor profile not found")
    
    # Get total sessions
    total_sessions = db.query(MentorshipSession).\
        filter(MentorshipSession.mentor_id == mentor_id).count()
    
    # Get active sessions
    active_sessions = db.query(MentorshipSession).\
        filter(MentorshipSession.mentor_id == mentor_id).\
        filter(MentorshipSession.status == Status.APPROVED).count()
    
    # Get completed sessions
    completed_sessions = db.query(MentorshipSession).\
        filter(MentorshipSession.mentor_id == mentor_id).\
        filter(MentorshipSession.status == Status.COMPLETED).count()
    
    # Get average rating
    reviews = db.query(Review).\
        join(User, Review.reviewee_id == User.id).\
        join(MentorProfile, User.id == MentorProfile.user_id).\
        filter(MentorProfile.id == mentor_id).all()
    
    avg_rating = sum(review.rating for review in reviews) / len(reviews) if reviews else 0
    
    return {
        "mentor_id": mentor_id,
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "completed_sessions": completed_sessions,
        "current_mentees": mentor.current_mentees_count,
        "max_mentees": mentor.max_mentees,
        "average_rating": avg_rating,
        "total_reviews": len(reviews)
    }

@app.get("/analytics/mentee-stats/{mentee_id}", tags=["Analytics"])
async def get_mentee_stats(mentee_id: str, db: Session = Depends(get_db)):
    """
    Get analytics for a specific mentee.
    """
    # Check if mentee exists
    mentee = db.query(MenteeProfile).filter(MenteeProfile.id == mentee_id).first()
    if mentee is None:
        raise HTTPException(status_code=404, detail="Mentee profile not found")
    
    # Get total sessions
    total_sessions = db.query(MentorshipSession).\
        filter(MentorshipSession.mentee_id == mentee_id).count()
    
    # Get active sessions
    active_sessions = db.query(MentorshipSession).\
        filter(MentorshipSession.mentee_id == mentee_id).\
        filter(MentorshipSession.status == Status.APPROVED).count()
    
    # Get completed sessions
    completed_sessions = db.query(MentorshipSession).\
        filter(MentorshipSession.mentee_id == mentee_id).\
        filter(MentorshipSession.status == Status.COMPLETED).count()
    
    # Get total meetings
    total_meetings = db.query(Meeting).\
        join(MentorshipSession, Meeting.session_id == MentorshipSession.id).\
        filter(MentorshipSession.mentee_id == mentee_id).count()
    
    # Get completed meetings
    completed_meetings = db.query(Meeting).\
        join(MentorshipSession, Meeting.session_id == MentorshipSession.id).\
        filter(MentorshipSession.mentee_id == mentee_id).\
        filter(Meeting.status == Status.COMPLETED).count()
    
    return {
        "mentee_id": mentee_id,
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "completed_sessions": completed_sessions,
        "total_meetings": total_meetings,
        "completed_meetings": completed_meetings,
        "expertise_level": mentee.expertise_level
    }

@app.get("/analytics/platform-stats/", tags=["Analytics"])
async def get_platform_stats(db: Session = Depends(get_db)):
    """
    Get overall platform analytics.
    """
    # Count users by role
    total_users = db.query(User).count()
    total_mentors = db.query(User).filter(User.role == UserRole.MENTOR).count()
    total_mentees = db.query(User).filter(User.role == UserRole.MENTEE).count()
    
    # Count sessions by status
    total_sessions = db.query(MentorshipSession).count()
    active_sessions = db.query(MentorshipSession).filter(MentorshipSession.status == Status.APPROVED).count()
    completed_sessions = db.query(MentorshipSession).filter(MentorshipSession.status == Status.COMPLETED).count()
    
    # Count meetings
    total_meetings = db.query(Meeting).count()
    completed_meetings = db.query(Meeting).filter(Meeting.status == Status.COMPLETED).count()
    
    # Average ratings
    avg_mentor_rating = db.query(func.avg(Review.rating)).\
        join(User, Review.reviewee_id == User.id).\
        filter(User.role == UserRole.MENTOR).scalar() or 0
    
    return {
        "user_stats": {
            "total_users": total_users,
            "total_mentors": total_mentors,
            "total_mentees": total_mentees
        },
        "session_stats": {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions
        },
        "meeting_stats": {
            "total_meetings": total_meetings,
            "completed_meetings": completed_meetings
        },
        "rating_stats": {
            "average_mentor_rating": avg_mentor_rating
        }
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
