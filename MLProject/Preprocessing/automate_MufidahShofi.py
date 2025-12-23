import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    """
    Fungsi untuk melakukan preprocessing data Hotel 
    
    """

    # LOAD DATA
    df = pd.read_csv(input_path)
    print("Data Loaded. Shape:", df.shape)

    # DROP ID COLUMN
    if 'Booking_ID' in df.columns:
        df = df.drop(columns=['Booking_ID']) 
    
    # HANDLE MISSING VALUES
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("Missing value handled.", df.shape)

    # REMOVE DUPLICATES
    df = df.drop_duplicates()
    print("Duplicates removed.", df.shape)


    # TARGET ENCODING
    df['booking_status'] = df['booking_status'].map({
        'Canceled': 1,
        'Not_Canceled': 0
    })

    # ONE-HOT ENCODING
    categorical_cols = [
        'type_of_meal_plan',
        'room_type_reserved',
        'market_segment_type'
    ]
    
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    # SCALING NUMERICAL FEATURES
    numerical_cols = [
        'lead_time',
        'avg_price_per_room',
        'no_of_adults',
        'no_of_children',
        'no_of_week_nights',
        'no_of_weekend_nights',
        'arrival_month',
        'arrival_date',
        'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled',
        'no_of_special_requests'
    ]

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # SAVE PROCESSED DATA
    df.to_csv(output_path, index=False)

    print("Preprocessing selesai!")
    print("Output saved to:", output_path)
    print("Final shape:", df.shape)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    input_path = os.path.join(
        ROOT_DIR, "Dataset-Raw", "hotel_reservations.csv"
    )

    output_path = os.path.join(
        BASE_DIR, "hotel_reservations_preprocessed.csv"
    )
    
    preprocess_data(input_path, output_path)