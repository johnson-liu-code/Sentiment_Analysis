import requests
import time

HEADERS = {'User-agent': 'CommentScraper/0.1'}
def get_top_comments(subreddit, post_limit=10, comment_limit=50):
    comments = []
    base_url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={post_limit}"

    try:
        res = requests.get(base_url, headers=HEADERS)
        posts = res.json()["data"]["children"]

        for post in posts:
            post_id = post["data"]["id"]
            post_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit={comment_limit}"
            time.sleep(1)  # prevent rate limiting

            try:
                comment_res = requests.get(post_url, headers=HEADERS)
                comment_data = comment_res.json()[1]["data"]["children"]
                for c in comment_data:
                    body = c.get("data", {}).get("body", "")
                    if body and body.lower() != "[removed]" and body.lower() != "[deleted]":
                        comments.append(body.replace("\n", " "))
            except:
                continue

    except Exception as e:
        print(f"❌ Couldn’t fetch from r/{subreddit}: {e}")

    return comments

def scrape_all_subreddits(subreddits, output_file="reddit_data.txt"):
    all_comments = []
    for sub in subreddits:
        print(f"📥 Scraping r/{sub}...")
        sub_comments = get_top_comments(sub)
        all_comments.extend(sub_comments)

    with open(output_file, "w", encoding="utf-8") as f:
        for comment in all_comments:
            f.write(comment + "\n")

    print(f"✅ Done! Saved {len(all_comments)} comments to {output_file}")

# 🔥 Subreddits to target
target_subs = ["memes", "sports", "drama", "movies", "cars", "anime"]

if __name__ == "__main__":
    scrape_all_subreddits(target_subs)
