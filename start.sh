#!/bin/bash

set -e

# Parse named flags
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --refreshPages)
      refreshPages="$2"
      shift 2
      ;;
    --graphPgL)
      graphPgL="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --genPgL)
      genPgL="$2"
      shift 2
      ;;
    --pageNums)
      pageNums="$2"
      shift 2
      ;;
    --title)
      title="$2"
      shift 2
      ;;
    --step)
      step="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

# Build final Python arg list
args1=()

[[ -n "$refreshPages" ]] && args1+=("--refreshPages" "$refreshPages")
[[ -n "$graphPgL" ]] && args1+=("--pageLimit" "$graphPgL")

args2=()

[[ -n "$config" ]] && args2+=("--config" "$config")
[[ -n "$genPgL" ]] && args2+=("--pageLimit" "$genPgL")
[[ -n "$pageNums" ]] && args2+=("--pageNums" "$pageNums")
[[ -n "$title" ]] && args2+=("--titleGeneration" "$title")
[[ -n "$step" ]] && args2+=("--step" "$step")




echo ">> Checking if everything is ready"
echo ">> Running: python3 init.py ${args1[*]}"
python3 init.py "${args1[@]}"
echo "All Set !"


echo ">> Running Voice generation steps"
echo ">> Running: python3 voiceGenerator.py ${args2[*]}"
python3 voiceGenerator.py "${args2[@]}"
echo "Done"