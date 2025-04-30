import requests
import time
import random
from datetime import datetime
import json

# Configuration
USER_AGENT = "script:reddit_scraper:v1.0 (by /u/YOUR_REDDIT_USERNAME)"
DELAY_BETWEEN_REQUESTS = 3  # seconds (minimum 2 recommended)
RETRY_DELAY = 30  # seconds when rate limited
MAX_RETRIES = 3
MAX_POSTS_PER_SUBREDDIT = 100  # Adjust based on your needs
MAX_COMMENTS_PER_POST = None  # Set to None for all comments, or a number to limit
OUTPUT_FILE = "reddit_comments_data.json"

# Proxy configuration (optional)
PROXY_LIST = [
    # "http://proxy1:port",
    # "http://proxy2:port",
    # Add more proxies if available
]

class RedditScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.scraped_data = {}
        
    def get_random_proxy(self):
        if PROXY_LIST:
            return {"http": random.choice(PROXY_LIST), "https": random.choice(PROXY_LIST)}
        return None
        
    def make_request(self, url):
        for attempt in range(MAX_RETRIES):
            try:
                proxies = self.get_random_proxy()
                response = self.session.get(url, proxies=proxies)
                
                # Check rate limits
                remaining = float(response.headers.get('X-Ratelimit-Remaining', 1))
                reset = float(response.headers.get('X-Ratelimit-Reset', 60))
                
                if remaining < 1:
                    wait_time = reset + random.uniform(0.5, 1.5)
                    print(f"Approaching rate limit. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = RETRY_DELAY * (attempt + 1) + random.uniform(0, 1)
                    print(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                print(f"HTTP Error: {e}")
                return None
            except Exception as e:
                print(f"Request failed: {e}")
                time.sleep(DELAY_BETWEEN_REQUESTS)
                continue
        return None
        
    def scrape_subreddit_posts(self, subreddit):
        base_url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        posts = []
        after = None
        
        print(f"\nScraping r/{subreddit}...")
        
        while len(posts) < MAX_POSTS_PER_SUBREDDIT:
            params = {"limit": 100}
            if after:
                params["after"] = after
                
            url = f"{base_url}?{requests.compat.urlencode(params)}"
            data = self.make_request(url)
            
            if not data:
                break
                
            new_posts = data["data"]["children"]
            posts.extend(new_posts)
            after = data["data"]["after"]
            
            if not after or len(new_posts) == 0:
                break
                
            print(f"  Collected {len(posts)} posts so far...")
            time.sleep(DELAY_BETWEEN_REQUESTS + random.uniform(0, 1))
            
        return posts[:MAX_POSTS_PER_SUBREDDIT]
        
    def scrape_post_comments(self, post):
        permalink = post["data"]["permalink"]
        comments_url = f"https://www.reddit.com{permalink}.json"
        
        data = self.make_request(comments_url)
        if not data or len(data) < 2:
            return []
            
        comments = []
        comment_list = data[1]["data"]["children"]
        
        for comment in comment_list:
            if comment["kind"] == "t1":  # Only process actual comments
                comment_data = comment["data"]
                comments.append({
                    "author": comment_data.get("author"),
                    "body": comment_data.get("body"),
                    "score": comment_data.get("score"),
                    "created_utc": comment_data.get("created_utc")
                })
                
                # Recursively get replies
                if "replies" in comment_data and comment_data["replies"] != "":
                    replies = self.extract_comments(comment_data["replies"]["data"]["children"])
                    comments.extend(replies)
        
        if MAX_COMMENTS_PER_POST:
            comments = comments[:MAX_COMMENTS_PER_POST]
            
        return comments
        
    def extract_comments(self, comment_objects):
        comments = []
        for comment in comment_objects:
            if comment["kind"] == "t1":
                comment_data = comment["data"]
                comments.append({
                    "author": comment_data.get("author"),
                    "body": comment_data.get("body"),
                    "score": comment_data.get("score"),
                    "created_utc": comment_data.get("created_utc")
                })
                if "replies" in comment_data and comment_data["replies"] != "":
                    replies = self.extract_comments(comment_data["replies"]["data"]["children"])
                    comments.extend(replies)
        return comments
        
    def scrape_subreddits(self, subreddits):
        for subreddit in subreddits:
            try:
                posts = self.scrape_subreddit_posts(subreddit)
                subreddit_data = []
                
                for i, post in enumerate(posts):
                    post_data = {
                        "title": post["data"].get("title"),
                        "url": f"https://www.reddit.com{post['data'].get('permalink')}",
                        "score": post["data"].get("score"),
                        "comments": []
                    }
                    
                    print(f"  Processing post {i+1}/{len(posts)}: {post_data['title'][:50]}...")
                    comments = self.scrape_post_comments(post)
                    post_data["comments"] = comments
                    subreddit_data.append(post_data)
                    
                    time.sleep(DELAY_BETWEEN_REQUESTS + random.uniform(0, 1))
                
                self.scraped_data[subreddit] = subreddit_data
                print(f"Finished r/{subreddit}. Collected {len(subreddit_data)} posts with comments.")
                
            except Exception as e:
                print(f"Error scraping r/{subreddit}: {e}")
                continue
                
    def save_data(self):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "date_scraped": datetime.now().isoformat(),
                    "subreddits": list(self.scraped_data.keys())
                },
                "data": self.scraped_data
            }, f, ensure_ascii=False, indent=2)
        print(f"\nData saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # List of subreddits to scrape
    SUBREDDITS = [
        "memes", "funny", "sports", "cars", "pokemon",
        "gaming", "movies", "music", "books", "food"
    ]
    
    scraper = RedditScraper()
    print(f"Starting Reddit scraper at {datetime.now()}")
    print(f"Scraping {len(SUBREDDITS)} subreddits with delay of {DELAY_BETWEEN_REQUESTS}s between requests")
    
    scraper.scrape_subreddits(SUBREDDITS)
    scraper.save_data()
    
    total_posts = sum(len(posts) for posts in scraper.scraped_data.values())
    total_comments = sum(len(post["comments"]) for posts in scraper.scraped_data.values() for post in posts)
    print(f"\nScraping complete! Collected {total_posts} posts with {total_comments} comments total.")