from IPython.display import display,HTML


def se(url=None):
    if url==None:
        url="https://ks-study.github.io/hosting/se.mp3"
    display(HTML("""
        <audio autoplay="" controls>
            <source src="{}"></source>
        </audio>
    """.format(url)))
