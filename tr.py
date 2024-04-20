import streamlit as st
import os
import urllib.parse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from dotenv import load_dotenv
from textblob import TextBlob
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Configure Google GenerativeAI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DEVELOPER_KEY = 'AIzaSyDYi0hx3ReDAlCz3GXom7hyj8t0vvjWcKs'  # Replace with your own developer key
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

def extract_video_id(url):
    query = urllib.parse.urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urllib.parse.parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    raise ValueError('Invalid YouTube URL or unable to extract video ID.')

def get_comments(video_id, max_results=100):
    try:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=max_results
        ).execute()

        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        return comments
    except HttpError as e:
        print(f'An error occurred: {e}')
        return []

def extract_comments(video_url):
    try:
        video_id = extract_video_id(video_url)
        comments = get_comments(video_id)
        return comments
    except ValueError as e:
        print(f'Error extracting comments: {e}')
        return []

# Function to perform sentiment analysis on comments
def analyze_sentiment(comments):
    sentiment_scores = []
    for comment in comments:
        comment_blob = TextBlob(comment)
        sentiment_scores.append(comment_blob.sentiment.polarity)  # Get polarity score for sentiment

    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    return average_sentiment

# Function to generate prompt for Gemini Pro based on sentiment analysis
def generate_prompt(sentiment_text):
    prompt = f"Based on the sentiment analysis of the comments, summarize the overall public opinion on the YouTube video with proper headings based on the opions of the public. Comments are mostly {sentiment_text}."
    return prompt

# Function to generate summary using Gemini Pro
def generate_summary(comments, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + " ".join(comments))
    return response.text

# Function to extract transcript details from a YouTube video URL
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript
    except Exception as e:
        raise e

# Streamlit UI
st.title('YouTube Video Analyzer')

# Get user input for YouTube video URL
video_url = st.text_input('Enter the YouTube video URL:', '')

# Button to trigger summary generation
if st.button("Generate Summary"):
    if video_url:
        # Extract transcript details
        transcript_text = extract_transcript_details(video_url)

        # Generate summary using Gemini Pro
        prompt = "Welcome, Video Summarizer! Your task is to distill the essence of a given YouTube video transcript into a concise summary with proper headings. Your summary should capture the key points and essential information, presented in bullet points, within a 250-word limit. Let's dive into the provided transcript and extract the vital details for our audience."
        summary = generate_summary(transcript_text, prompt)
        st.markdown("## Summary of Transcript:")
        st.write(summary)
        
        # Store the generated summary in session state
        st.session_state['transcript_summary'] = summary

# Button to trigger comment analysis
if st.button("Analyze Comments"):
    if video_url:
        # Extract comments
        comments = extract_comments(video_url)
        if comments:
            st.success('Comments extracted successfully!')
            st.write('Number of comments extracted:', len(comments))

            # Perform sentiment analysis
            average_sentiment = analyze_sentiment(comments)
            sentiment_text = "positive" if average_sentiment > 0 else "negative" if average_sentiment < 0 else "neutral"
            st.write(f"Overall sentiment of the comments: {sentiment_text}")

            # Generate prompt based on sentiment
            prompt = generate_prompt(sentiment_text)

            # Generate summary using Gemini Pro
            comments_summary = generate_summary(comments, prompt)
            st.markdown("## Summary of Comments:")
            st.write(comments_summary)
            
            # Store the generated summary in session state
            st.session_state['comments_summary'] = comments_summary

# Conditionally display the transcript summary if it exists
if 'transcript_summary' in st.session_state:
    st.markdown("## Summary of Transcript:")
    st.write(st.session_state['transcript_summary'])

# Button to generate blog article
if st.button("Generate Blog"):
    if 'transcript_summary' in st.session_state and 'comments_summary' in st.session_state:
        # Merge summaries
        merged_summary = f"Summary of Transcript:\n\n{st.session_state['transcript_summary']}\n\nSummary of Comments:\n\n{st.session_state['comments_summary']}"

        # Generate blog article using Gemini AI
        prompt = "Based on the summary of the transcript and comments, generate a blog article discussing the key insights and public sentiment of the YouTube video. Your article should provide a comprehensive overview while maintaining a neutral tone."
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + merged_summary)
        blog_article = response.text

        # Display generated blog article
        st.markdown("## Generated Blog Article:")
        st.write(blog_article)
