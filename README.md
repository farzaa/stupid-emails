# stupid-emails
A simple Python script to tag your G-Mail with AI.


1. `git clone https://github.com/farzaa/stupid-emails.git`
2. `cd stupid-emails`
2. Rename `secrets.template.json` to `secret.json` and add your OpenAI key in to it.
3. Go to `prompt.txt` and add your name + bio.
4. `pip install -r requirements.txt`
5. Run script `python tag.py`, Google Auth will open and ask for read/write acceess. The script takes your email, send it to OpenAI, and gets back a tag. That's it. Nothing is saved. It's not gonna randomly send emails on your behalf lol. You can read through the code if you don't trust me.
