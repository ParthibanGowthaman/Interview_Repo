# Interview Coding Challenge - Forecast Quarter Enhancement

## Overview
You have been provided with a warehouse asking rent forecasting script that currently only supports yearly forecast ranges. Your task is to enhance it to support quarterly granularity.

## The Challenge
The script `forecast_with_quarters_challenge.py` has several bugs and missing implementations marked with:
- `# BUG #X:` - Identifies the issue
- `# TODO:` - Indicates what needs to be implemented

### Main Tasks:
1. Add support for forecast start and end quarters (Q1-Q4)
2. Implement quarter-to-month conversion logic
3. Update date filtering to respect quarter boundaries
4. Fix all related bugs throughout the script

### Key Requirements:
- Quarters map to months as: Q1=Jan(1), Q2=Apr(4), Q3=Jul(7), Q4=Oct(10)
- Forecast should start from the specified year AND quarter
- Forecast should end at the specified year AND quarter
- All displays and outputs should reflect the quarter information

## Instructions

### Setup
1. Fork this repository to your GitHub account
2. Clone your forked repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Interview_Repo.git
   cd Interview_Repo
   ```

3. Create a new branch:
   ```bash
   git checkout -b fix/your-name
   ```

### Fixing the Code
1. Open `forecast_with_quarters_challenge.py`
2. Search for all `BUG` comments to understand what needs fixing
3. Implement the missing functionality
4. Test your implementation logic

### Submission
1. Commit your changes:
   ```bash
   git add .
   git commit -m "Fix: Add quarter support to forecast date range"
   ```

2. Push to your fork:
   ```bash
   git push origin fix/your-name
   ```

3. Create a Pull Request from your fork to this repository

## Time Limit
You have **45 minutes** to complete this challenge.

## Evaluation Criteria
- **Correctness**: All bugs are fixed and quarter logic works properly
- **Code Quality**: Clean, readable, and maintainable code
- **Edge Cases**: Handles boundary conditions correctly
- **Git Practices**: Clear commit messages and PR description

## Hints
- Pay attention to how the data uses months 1, 4, 7, 10 for quarters
- Consider edge cases like starting in Q3 of one year and ending in Q2 of another
- Make sure all references to the forecast period are updated consistently

## Need Help?
If you're stuck on understanding the requirements (not the solution), feel free to ask for clarification.

Good luck!