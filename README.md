## ChatBot with LLMs

<img width="100%" alt="Screenshot 2024-01-29 at 3 55 38 PM" src="public/demo.png">

#### Output comparison from various models

<img width="80%" alt="Screenshot 2024-01-29 at 3 55 38 PM" src="public/deepseek-r1.png">

<img width="80%" alt="Screenshot 2024-01-29 at 3 55 38 PM" src="public/llama2.png">

### How to run the app

1. Setup virtual environment with [`pipenv`](https://pipenv.pypa.io/en/latest/installation.html)

2. Install dependencies

   ```
   pipenv install
   ```

   To install exact versions run:

   ```
   pipenv install --ignore-pipfile
   ```

3. Download ollama (Two approaches:)

   - form from [github](https://github.com/ollama/ollama?tab=readme-ov-file)
     or

   - from the [website](https://ollama.com/download/mac)

4. Download LLM models via ollama

   Explore list of LLM models via ollama [here](https://ollama.com/library)

   e.g. to download `llama3.2` run:

   ```
   ollama run llama3.2
   ```

   e.g. to download `deepseek-r1` run:

   ```
   ollama run deepseek-r1
   ```

   To download model with specific size e.g. 14B parameters = 9.0GB here is the [link](https://ollama.com/library/deepseek-r1:14b)

   ```
   ollama run deepseek-r1:14b
   ```

5. Run the app

   ```
   python app.py
   ```

   OR

   ```
   python3 app.py
   ```

### App challanges

- live rendering of outputs to front end (frontend problems)
- some models are rellay heavy, but accurate (smller models are less acurate)
- formatting of outputs e.g tables, codes, images
- performing quantitative analysis and visualizations & rendiering those to front end

### some goals

- add a drop down menu to select model
- format and clean generated quntitatve data to perform visualizations
- steering LLM output using feedback mechanism
