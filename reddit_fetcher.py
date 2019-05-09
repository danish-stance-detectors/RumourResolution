from datetime import datetime as dt

def getredditsubmission(reddit, subid):
    """Retrieve information in JSON about a Submission, inlcuding its
    author, Subreddit, and all comments with corresponding user unfo"""
    submission = reddit.submission(id=subid) 

    sub_data = submissioninfo(submission)
    sub_data['user'] = userinfo(submission.author)
    sub_data['subreddit'] = subredditinfo(submission.subreddit, submission.subreddit_id)
    submission.comment_sort = 'old'
    sub_data['comments'] = commentsinfo(submission.comments)
    return sub_data


def submissioninfo(submission):
    """Retrieve essential data for a Submission"""
    sub_data = {}
    sub_data['title'] = submission.title
    sub_data['text'] = submission.selftext
    sub_data['submission_id'] = submission.id
    sub_data['created'] = utctodate(submission.created_utc)
    sub_data['num_comments'] = submission.num_comments
    sub_data['url'] = submission.permalink
    sub_data['text_url'] = submission.url
    sub_data['upvotes'] = submission.score
    sub_data['is_video'] = submission.is_video
    return sub_data

def userinfo(user):
    """Retrieve relevant information for a Redditor"""
    if ( user is None ):
        return {}
    user_data = {}
    try:
        user_data['id'] = user.id
        user_data['username'] = user.name
        user_data['karma'] = user.comment_karma
        user_data['created'] = utctodate(user.created_utc)
        user_data['gold_status'] = user.is_gold
        user_data['is_employee'] = user.is_employee
        user_data['has_verified_email'] = user.has_verified_email if user.has_verified_email is not None else False
    except Exception:
        return user_data
    return user_data

def subredditinfo(subreddit, subreddit_id):
    """Retrieve essential data for a Subreddit"""
    subreddit_data = {}
    subreddit_data['name'] = subreddit.display_name
    subreddit_data['subreddit_id'] = subreddit_id
    subreddit_data['created'] = utctodate(subreddit.created_utc)
    subreddit_data['subscribers'] = subreddit.subscribers
    return subreddit_data

def commentsinfo(comments):
    comments_data = []
    while True:
        try:
            #all MoreComments objects will be replaced
            #May cause many API calls, and thus exceptions
            #Keep trying until all are replaced
            comments.replace_more(limit=None)
            break
        except Exception:
            print('Handling replace_more exception')
            time.sleep(1)
    for comment in comments.list(): #flatten all nested comments
        data = {}
        try:
            data['comment_id'] = comment.id
            data['text'] = comment.body
            is_deleted = False
            if ( data['text'] == '[deleted]'):
                is_deleted = True
            data['is_deleted'] = is_deleted
            data['created'] = utctodate(comment.created_utc)
            data['is_submitter'] = comment.is_submitter
            data['submission_id'] = comment.link_id
            data['parent_id'] = comment.parent_id
            data['comment_url'] = comment.permalink
            data['upvotes'] = comment.score
            data['replies'] = comment.replies.__len__()
            data['user'] = userinfo(comment.author)
        except Exception:
            print("Comment fail")

        comments_data.append(data)
    return comments_data

def utctodate(utctime):
    """Convert POSIX time to YYYY-MM-DD HH:MM:SS"""
    return dt.utcfromtimestamp(utctime).strftime("%Y-%m-%d %H:%M:%S")

def datetoutc(date):
    """Convert date in  list format [YYYY, M, D] to POSIX time"""
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    return int(dt(year, month, day).timestamp())