# utils.py - Utility functions for configuration and S3 operations
import os
import boto3
import configparser

s3 = boto3.client('s3')


def load_certificate_from_s3(bucket_name, file_name):
    """Load certificate from S3 based on bucket and file name."""
    s3_response = s3.get_object(Bucket=bucket_name, Key=file_name)
    pem_content = s3_response['Body'].read()
    return pem_content


def load_config():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'config.ini')
    llm_config = configparser.ConfigParser()
    llm_config.read(file_path)
    return llm_config