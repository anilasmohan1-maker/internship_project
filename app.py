StopIteration: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/internship_project/app.py", line 148, in <module>
    summary = generator.resume_summary(profile)
File "/mount/src/internship_project/app.py", line 41, in resume_summary
            return self.chat(
                   ~~~~~~~~~^
                system_prompt="You are a professional resume writer.",
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<7 lines>...
    """
    ^^^
            )
            ^
File "/mount/src/internship_project/app.py", line 30, in chat
    response = client.chat_completion(
        messages=[
    ...<4 lines>...
        temperature=0.7
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/huggingface_hub/inference/_client.py", line 878, in chat_completion
    provider_helper = get_provider_helper(
        self.provider,
    ...<3 lines>...
        else payload_model,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/huggingface_hub/inference/_providers/__init__.py", line 217, in get_provider_helper
    provider = next(iter(provider_mapping)).provider
               ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
