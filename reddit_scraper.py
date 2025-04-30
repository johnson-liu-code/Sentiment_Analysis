import requests
import datetime
import time

subreddits = ["memes", "funny", "sports", "cars", "pokemon", "gaming", "movies", "music", "books", "food"]
post_limit = 10
max_comments_per_post = 5

headers = {
    "User-Agent": "script:myreddittape:v1.0 (by /u/YourRedditUsername)"
}

file_path = "reddit_data.txt"

def scrape_comments_only():
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"Scraper started at {datetime.datetime.now()}\n\n")
            
            for subreddit in subreddits:
                print(f"Scraping r/{subreddit}...")
                posts_url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={post_limit}"
                
                try:
                    response = requests.get(posts_url, headers=headers)
                    response.raise_for_status()
                    
                    # FIX: Handle float values in X-Ratelimit-Remaining (e.g., '99.0')
                    remaining_requests = int(float(response.headers.get('X-Ratelimit-Remaining', 1)))
                    reset_time = float(response.headers.get('X-Ratelimit-Reset', 60))
                    
                    if remaining_requests < 1:
                        print(f"Rate limited. Waiting {reset_time + 1} seconds...")
                        time.sleep(reset_time + 1)
                    
                    data = response.json()
                    
                    for post in data["data"]["children"]:
                        permalink = post["data"].get("permalink")
                        if not permalink:
                            continue
                        
                        comments_url = f"https://www.reddit.com{permalink}.json"
                        try:
                            comments_response = requests.get(comments_url, headers=headers)
                            comments_response.raise_for_status()
                            comments_data = comments_response.json()
                            
                            post_title = post["data"].get("title", "No Title")
                            file.write(f"\nPost Title: {post_title}\n")
                            file.write(f"Link: https://www.reddit.com{permalink}\n")
                            file.write("Comments:\n")
                            
                            comments = comments_data[1]["data"]["children"]
                            comment_count = 0
                            
                            for comment in comments:
                                if comment["kind"] != "t1":
                                    continue
                                comment_body = comment["data"].get("body", "[deleted]")
                                file.write(f"- {comment_body}\n")
                                comment_count += 1
                                if comment_count >= max_comments_per_post:
                                    break
                            
                            file.write("\n" + "-" * 40 + "\n\n")
                            time.sleep(2)  # Delay between comment fetches
                            
                        except Exception as e:
                            print(f"Error fetching comments for post: {e}")
                            continue
                    
                    time.sleep(5)  # Delay between subreddits
                    
                except Exception as e:
                    print(f"Error scraping r/{subreddit}: {e}")
                    file.write(f"Error scraping r/{subreddit}: {e}\n")
                    continue
    
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    scrape_comments_only()
    print("Scraping completed. Data saved to", file_path("reddit_data.txt"))