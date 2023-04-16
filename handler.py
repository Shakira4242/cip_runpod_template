import runpod
import os
import time
import app as user_src

## load your model(s) into vram here
user_src.init()

def handler(job):
    job_input = job["input"]
    image = job_input["image"]
    return user_src.inference(image)

runpod.serverless.start({
    "handler": handler
})