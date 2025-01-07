import streamlit as st
import pandas as pd

def main():
    # Title for the Streamlit app
    st.title("Peer Assessment Tricks")

    # First text input field for names
    names_input = st.text_input("Enter names (comma-separated):")

    # Process the input if it's not empty
    if names_input:
        all_members = [name.strip() for name in names_input.split(",")]

    # Second text input field for another set of values
    values_input = st.text_input("Enter other values (comma-separated):")

    # Process the input if it's not empty
    if values_input:
        categories = [value.strip() for value in values_input.split(",")]

    # Create two columns
    col1, col2 = st.columns(2)

    # Left column: Slider for range selection
    with col1:
        min_value, max_value = st.slider(
            "Select a range (0 to 100):",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=1,
        )
        st.write(f"Selected Range: {min_value} to {max_value}")

    # Right column: Slider for step size
    with col2:
        step_size = st.number_input(
            "Insert a step size (1 to 10):",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )
        st.write(f"Selected Step Size: {step_size}")

    # Generate the range as a list
    if min_value < max_value:
        marks = list(range(min_value, max_value + 1, step_size))

    if names_input and values_input:
        # Marks for each iteration
        # marks = [4, 3, 2, 1]  # Distinct marks for each category

        # Store total marks for all members
        total_marks = {member: 0 for member in all_members}

        # Create a DataFrame with group members as index
        summary_df = pd.DataFrame({"Name of Group Member":all_members}).set_index("Name of Group Member")

        # Generate tables for 5 iterations
        for iteration in range(1, 6):
            data = []

            # Exclude one member for this iteration
            excluded_member = all_members[(iteration - 1) % len(all_members)]
            members = rotated_list = all_members[iteration:] + all_members[:iteration]
            members = [member for member in members if member != excluded_member]

            # Distribute marks ensuring unique values in each column
            for i, member in enumerate(members):
                member_marks = {}
                for j, category in enumerate(categories):
                    member_marks[category] = marks[(i + j) % len(marks)]

                # Add total marks
                member_marks[excluded_member] = sum(member_marks[category] for category in categories)
                total_marks[member] += member_marks[excluded_member]
                data.append(member_marks)

            # Create a DataFrame for the current iteration
            df = pd.DataFrame(data, index=members)
            df.index.name = "Name of Group Member"

            # Write to a sheet in the Excel file
            sheet_name = excluded_member
            st.write(f"{sheet_name}'s assessment:")
            st.dataframe(df)
            st.divider()

            # Create a summary_df with total marks for all members
            summary_df = summary_df.merge(df.loc[:,excluded_member], how="left", left_index=True, right_index=True)

        # Add the total marks for all members to the summary_df
        summary_df.loc[:,"Total Marks"] = summary_df.sum(axis=0)
        
        # Display the summary_df
        st.write("Summary:")
        st.dataframe(summary_df)

if __name__ == "__main__":
    main()