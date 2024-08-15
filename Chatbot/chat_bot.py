import random

exit_words = ["bye", "thank you for your time", "take care", "bye for now", "have a great day", "all the best", "keep in touch"]


#select the topic related to the conversation
def get_topics():
    topic = {
        'tech' : ['technology', 'tech', 'ai', 'llm', 'google'," hardware", 'software', 'networking', 'development', 'cloud', 'security', 'data', 'microsoft', 'mobile', 'nvidia', 'amd', 'artificial intelligence', 'chatgpt', 'gpt', 'generative ai'],

        'edu' : ['curriculum','syllabus','edu','education','pedagogy', 'assessment', 'evaluation','primary', 'secondary','higher', 'vocational', 'teacher', 'professor', 'tutor', 'classromm', 'laboratory', 'lecture', 'textbook', 'lecture', 'exam','quiz', 'project', 'books', 'book', 'pen', 'paper'], 
 
        'social_media' : ['post', 'social', 'social media', 'share', 'like', 'follow', 'comment', 'tag', 'hashtag', 'feed', 'timeline', 'profile', 'story', 'reel', 'livestrem', 'viral', 'mention', 'emoji', 'follower', 'subscriber','notification', 'platform', 'facebook', 'x', 'twitter', 'linkedin', 'instagram', 'whatsapp', 'telegram', 'wechat','tweet', 'retweet', 'community'],
    }
    return topic



def exit_the_conversation(text):
    for words in exit_words:
        if words in text:
            print("Thank you for your time, If you have any query then let me know..!")
            return True   



def search_topic(txt):
    topics = get_topics()

    for topic, elements in topics.items():
        for k in elements:
            if k in txt:
                topic_name = topic
                return topic_name
                exit(1)
       


def technology():
    info = ['Technology has completely changed how we communicate. With computers and cellphones, we can instantaneously communicate with friends and family who live all over the world.We can also share our lives and hobbies with a large audience through social media platforms.',
            
    'Education has changed as a result of technology. Online learning environments provide accessible and adaptable educational opportunities. Students have access to a wealth of knowledge and can study at their own speed',

    'Technology has impacted the entertainment industry tremendously. Access to a vast array of films, TV series, and music is possible via streaming services. Immersive gaming and entertainment experiences are provided via virtual and augmented reality.',

    'Healthcare has improved thanks to technology. More accurate disease diagnosis and treatment are made possible by medical software and technology. Healthcare is now more accessible because to telemedicine, which enables patients to consult with physicians at a distance.',

    'We now live more convenient lives thanks to technology. Time and energy are saved by automating chores in smart products and homes. Product delivery services and online shopping bring goods directly to our front door.'
    ]

    print(random.choice(info))



def education():
    info = ['Lifelong learning is a journey. We receive early guidance from schools and teachers, who aid in the development of our math, reading, and writing abilities. But education does not end there.',
            
    'Being curious is a useful skill.Anything that piques our curiosity can be learned about via books, the internet, and practical experience. Finding our passions is facilitated by trying new things and engaging in hobbies.',

    'Errors are a natural part of learning. We learn and improve our problem-solving abilities through challenges. It is critical to maintain your resolve and perseverance in the face of difficulty.',

    'Everybody learns in a unique way. While some people learn best by doing, others prefer visual aids. To maximize your educational experience, identify your preferred method of learning.',

    'Knowledge gives us power. The ability to make informed judgments and accomplish our objectives is provided by knowledge. Having a good education gives us more options and makes us effective adults.'
    ]

    print(random.choice(info))


def social_media():
    info = ['Our relationships with friends and family have evolved as a result of social media. Sharing films, pictures, and updates with distant relatives is simple. We might also run across new folks who share our interests.',
            
    'Being polite on the internet is crucial. Everyone is entitled to respectful treatment. Online communities can be made happier by promoting positivism and offering assistance to others.',

    'Using social media to express oneself can be enjoyable. You can find your talents by showcasing your creative through blogs, tales, or videos.',

    'Think twice before sharing anything online. Not every individual is who they claim to be. Keep your personal information safe and exercise caution when posting.',

    'While it is simple to become engrossed in social media, it is crucial to balance screen time with other activities. It is also crucial to play outside, engage in hobbies, and spend time with loved ones.'
    ]

    print(random.choice(info))





def start_chat():
    print()
    res = input("Tell me the topic that you want to discuss, Remember! I can only discuss Technology, Education or Social Media related topics\n").lower()
 
    if exit_the_conversation(res):
        return True
    
    topic_name = search_topic(res)
    
    if topic_name == 'tech':
        technology()
        start_chat()
    elif topic_name == 'edu':
        education()
        start_chat()
    elif topic_name == 'social_media':
        social_media()
        start_chat()
    else:
        print("Sorry! I do not find the topic. Ask about Technology, Education or Social media related topics. \n")
        print()
        start_chat()



def start_conversation():
    res = input("Hello, I am a bot, what is your name?\n")
    res = input(f"Hello {res}, How can a help you today? \n")
    if exit_the_conversation(res):
        return True
    
    start_chat()



start_conversation()