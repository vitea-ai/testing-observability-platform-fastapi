import asyncio
import pandas as pd
import json

async def test_parser():
    # Read the CSV file
    df = pd.read_csv('test_dataset.csv')
    print("Columns:", df.columns.tolist())
    print("\nFirst row:")
    print(df.iloc[0].to_dict())
    print("\nData shape:", df.shape)
    
    # Check if it's conversation format
    if 'conversation_id' in df.columns:
        print("\nConversation format detected!")
        # Group by conversation_id
        for conv_id, group in df.groupby('conversation_id'):
            print(f"\nConversation: {conv_id}")
            for idx, row in group.iterrows():
                print(f"  Input: {row['input']}")
                print(f"  Output: {row['output']}")
                print(f"  Expected: {row['expected_output']}")
                if 'metadata' in row and pd.notna(row['metadata']):
                    print(f"  Metadata: {row['metadata']}")
            break  # Just show first conversation

asyncio.run(test_parser())
