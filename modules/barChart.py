import matplotlib.pyplot as plt
import io


def generate_co2_chart(data):
    """
    Generates a bar chart from CO2 emissions data and returns an image file.

    Parameters:
        data (list): A list of dictionaries containing State and Emission values.

    Returns:
        io.BytesIO: The image file in memory.
    """
    # Extract states and values
    states = [item["State"] for item in data]
    values = [item["amount_value"] for item in data]

    # Create bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(states, values, color='blue')

    # Labels and title
    plt.xlabel("States")
    plt.ylabel("CO2 Emissions (in metric tons)")
    plt.title("CO2 Emissions by State")
    plt.xticks(rotation=45, ha="right")

    # Save the image to a BytesIO buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)  # Move to the beginning of the file

    plt.close()  # Close the plot to free memory
    return img_buffer
