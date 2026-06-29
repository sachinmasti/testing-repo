from pydantic import BaseModel,field_validator,Field
from  typing import Annotated,Literal

data = ['age', 'gender', 'daily_social_media_hours', 'platform_usage',
       'sleep_hours', 'screen_time_before_sleep', 'academic_performance',
       'physical_activity', 'social_interaction_level', 'stress_level',
       'anxiety_level', 'addiction_level', 'depression_label']



class Info(BaseModel):
    age : Annotated[int,Field(...,description='enter a age of student',gt=12,lt=20)]
    gender : Literal['male','female']
    daily_social_media_hours : Annotated[float,Field(...,description='enter your daily social media usage in hours')]
    platform_usage : Literal['Instagram','TikTok','Both']
    sleep_hours : Annotated[float,Field(...,description='enter your sleep hours')]
    screen_time_before_sleep : Annotated[float,Field(...,description='enter your screen time before sleep')]
    physical_activity : Annotated[float,Field(...,description='enter your physical activity')]
    social_interaction_level : Literal['low','medium','high']
    stress_level : Annotated[int,Field(...,description='enter your stress level',examples=[1,3,4])]
    anxiety_level : Annotated[int,Field(...,description='enter your anxiety level',examples=[1,6,8])]
    addiction_level : Annotated[int,Field(...,description='enter a addiction level of the student',examples=[1,5,7])]
    depression_label: Annotated[int, Field(..., description='enter depression label',examples=['yes','no',1,0])]

    @field_validator('age')
    @classmethod
    def age_validate(cls,value):
        if value < 12 or value > 20:
            raise ValueError ('enter a age between 12-20')
        return value

    @field_validator('gender',mode='before')
    @classmethod
    def valid_gender(cls, value):
        return value.lower()

    @field_validator('social_interaction_level',mode='before')
    @classmethod
    def valid_socials(cls, value):
        return value.lower()

    @field_validator('depression_label',mode='before')
    @classmethod
    def depression_label_int(cls,value):
        if isinstance(value,str):
            value = value.lower()
            return 1 if value == 'yes' else 0
        return value







#
# def test(details:Info):
#     print(details)
#
# test(Info(
#     age=15,
#     gender='Male',
#     daily_social_media_hours=3.5,
#     platform_usage='TikTok',
#     sleep_hours=7.0,
#     screen_time_before_sleep=1.5,
#     physical_activity=2.0,
#     social_interaction_level='Medium',
#     stress_level=5,
#     anxiety_level=4,
#     addiction_level=3,
#     depression_label='no'
# ))
