#!/bin/bash

# Utility script for encrypting and decrypting files
set -e

COMMAND=$1
KEY=$2
ENCRYPTED_FILE="env.encrypted"
DECRYPTED_FILE=".env"

# Check for openssl
if ! command -v openssl &> /dev/null; then
  echo "Error: openssl is not installed. Please install it and try again."
  exit 1
fi

# Check if the key is provided
if [[ -z "$KEY" ]]; then
  echo "Usage: $0 <encrypt|decrypt> <encryption-key>"
  exit 1
fi

case "$COMMAND" in
  encrypt)
    if [[ -f "$DECRYPTED_FILE" ]]; then
      echo "Encrypting $DECRYPTED_FILE to $ENCRYPTED_FILE..."
      openssl enc -aes-256-cbc -salt -pbkdf2 -in "$DECRYPTED_FILE" -out "$ENCRYPTED_FILE" -k "$KEY"
      echo "Encryption successful: $ENCRYPTED_FILE created."
    else
      echo "Error: $DECRYPTED_FILE not found."
      exit 1
    fi
    ;;
  decrypt)
    if [[ -f "$ENCRYPTED_FILE" ]]; then
      echo "Decrypting $ENCRYPTED_FILE to $DECRYPTED_FILE..."
      openssl enc -aes-256-cbc -d -pbkdf2 -in "$ENCRYPTED_FILE" -out "$DECRYPTED_FILE" -k "$KEY"
      echo "Decryption successful: $DECRYPTED_FILE created."
    else
      echo "Error: $ENCRYPTED_FILE not found."
      exit 1
    fi
    ;;
  *)
    echo "Invalid command. Usage: $0 <encrypt|decrypt> <encryption-key>"
    exit 1
    ;;
esac