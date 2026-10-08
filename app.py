from pathlib import Path
import json
import zipfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'assets'
# Community Cloud's browser-upload package contains the same verified assets in a ZIP.
if not DATA.exists():
    with zipfile.ZipFile(ROOT / 'demo_assets.zip') as archive:
        for member in archive.infolist():
            target = (ROOT / member.filename).resolve()
            if not member.filename.startswith('assets/') or not target.is_relative_to(ROOT):
                raise ValueError('Invalid bundled asset path')
        archive.extractall(ROOT)
st.set_page_config(page_title='H.E.L.E.N | TCN vs LSTM', page_icon='🤟', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1220;}
[data-testid="stMetric"] {background: #162238; padding: 18px; border-radius: 12px;}
.block-container {padding-top: 2rem; max-width: 1400px;}
h1,h2,h3 {letter-spacing: -.025em;}
[data-testid="stAlert"] p {color: #e7edf7;}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load():
    auslan = json.loads((DATA / 'auslan_results.json').read_text())
    weather = pd.read_csv(DATA / 'climate_predictions.csv')
    metrics = pd.read_csv(DATA / 'climate_comparison.csv')
    return auslan, weather, metrics

auslan, weather, metrics = load()
st.sidebar.title('H.E.L.E.N')
st.sidebar.caption('Sequence learning • weekly progress demo')
page = st.sidebar.radio('Explore', ['Overview', 'Auslan sign recognition', 'Kaggle climate forecast', 'Understand TCN & LSTM', 'Our ISL project', 'Sources & reproduction'])
st.sidebar.divider()
st.sidebar.caption('Previously trained models • real recorded results\n\nThis dashboard does not train models or recognize webcam signs online.')

def plot(fig):
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0b1220', plot_bgcolor='#0b1220', margin=dict(t=45,b=25,l=25,r=25), legend=dict(orientation='h', y=-.2))
    st.plotly_chart(fig, width='stretch')

def summary():
    c = st.columns(4)
    c[0].metric('Auslan recordings', '699', '10 signs · 5 signers', delta_color='off')
    c[1].metric('Held-out test accuracy', '49.69%', 'LSTM · TCN 30.19%', delta_color='off')
    c[2].metric('Climate forecast days', '115', '4 final model configurations', delta_color='off')
    c[3].metric('Best local climate RMSE', '1.849 °C', 'LSTM · temperature only', delta_color='off')

if page == 'Overview':
    st.caption('H.E.L.E.N / EXPERIMENT LAB')
    st.title('Two models. Two real experiments.')
    st.write('We trained LSTM and Temporal Convolutional Network (TCN) models to study how they learn from sequences. These experiments establish a comparison workflow before collecting Indian Sign Language (ISL) recordings.')
    summary()
    left, right = st.columns(2)
    with left:
        st.subheader('01 · Recognize a recorded sign')
        st.write('30 timesteps of glove movement → one of 10 Auslan words. Both models saw the same training recordings; the test person was held out.')
        st.info('LSTM performed better in this run. These are Australian signs recorded by a glove, not Indian signs or camera video.')
    with right:
        st.subheader('02 · Predict the next temperature')
        st.write('30 previous daily observations → next-day temperature. We reproduced Ricardo Colindres’s Kaggle notebook with temperature alone and temperature plus humidity.')
        st.info('The local best scores were very close. LSTM achieved 1.849 °C RMSE; the best TCN achieved 1.855 °C.')
    st.subheader('The progress we can demonstrate today')
    st.markdown('1. Use existing datasets and prepare sequences.\n2. Train both architectures and preserve their outputs.\n3. Compare errors, accuracy, runtime and failure cases.\n4. Use the same workflow for verified ISL recordings next.')
    st.warning('TCN is a candidate for H.E.L.E.N. We have not demonstrated that it beats LSTM on ISL. Neither experiment implements bidirectional sign-to-speech communication yet.')

elif page == 'Auslan sign recognition':
    st.title('01 · Auslan sign recognition')
    st.write('Real UCI PowerGlove recordings: 8 channels per timestep, resampled to 30 timesteps. A model predicts a word from the entire movement sequence.')
    models = auslan['models']
    table = pd.DataFrame([{'Model': n, 'Accuracy (%)': m['accuracy']*100, 'Macro F1': m['macro_f1'], 'Parameters': m['parameters'], 'Training (s)': m['training_seconds'], 'Median inference (ms)': m['latency_median_ms']} for n,m in models.items()])
    st.dataframe(table.round(3), hide_index=True, width='stretch')
    st.caption('Inference times are batch=1 on the local CPU and exclude camera capture, landmark extraction and speech. Model sizes and architectures are different; this is one baseline run.')
    tabs = st.tabs(['Explore a recording', 'Confusion matrices', 'Learning & evaluation'])
    with tabs[0]:
        samples = auslan['data_metadata']['sample_previews']
        idx = st.selectbox('Recorded test sample', range(len(samples)), format_func=lambda i: f"{samples[i]['label']} · {samples[i]['file']}")
        sample = samples[idx]
        st.subheader(f"Actual word: {sample['label']}")
        cols = st.columns(2)
        for col,name in zip(cols,['LSTM','TCN']):
            p = sample['predictions'][name]
            col.metric(name, p['label'], f"Confidence {p['confidence']:.1%}", delta_color='off')
        trace = pd.DataFrame(sample['trace'], columns=['x','y','z'])
        trace['Timestep'] = range(1,len(trace)+1)
        plot(px.line(trace, x='Timestep', y=['x','y','z'], title='Recorded hand position through time'))
        st.caption('Three position channels are shown; the classifier uses all eight channels. Confidence is a model score, not a calibrated probability of correctness. These 20 previews were selected independently of correctness.')
    with tabs[1]:
        name = st.selectbox('Model', ['LSTM','TCN'])
        fig = px.imshow(models[name]['confusion_matrix'], x=auslan['labels'], y=auslan['labels'], text_auto=True, labels={'x':'Predicted word','y':'Actual word','color':'Recordings'}, aspect='auto')
        plot(fig)
        st.write('The diagonal contains correct predictions. Off-diagonal cells reveal which words the model confused. Macro F1 averages the score of each word equally.')
    with tabs[2]:
        fig = go.Figure()
        for name,m in models.items():
            fig.add_scatter(x=list(range(1,len(m['validation_loss'])+1)), y=m['validation_loss'], name=name)
        fig.update_layout(title='Validation loss by epoch', xaxis_title='Epoch', yaxis_title='Cross-entropy loss')
        plot(fig)
        st.write('Train: John, Waleed and Adam (460 recordings). Validation: Andrew (80). Test: Stephen (159). Keeping people separate measures transfer to an unseen signer.')
        st.write('Both models used Adam, learning rate 0.001, batch size 16 and a maximum of 100 epochs. The lowest validation-loss checkpoint was evaluated on the test signer.')
        st.error('TCN accuracy: 30.19%; LSTM accuracy: 49.69%. This baseline needs improvement before practical sign recognition.')

elif page == 'Kaggle climate forecast':
    st.title('02 · Kaggle climate forecast')
    st.write('Daily Delhi climate data: 1,576 combined records. The notebook learns from the years before 2017 and evaluates 115 records in 2017. This is numerical forecasting, not sign recognition.')
    mode = st.radio('Forecast mode', ['One-step: actual previous observations', 'Recursive: feed predictions back'], horizontal=True)
    if mode.startswith('One-step'):
        features = st.selectbox('Input features', ['Temperature only','Temperature + humidity'])
        suffix = 'one_ft' if features == 'Temperature only' else 'two_ft'
        st.write('Each prediction uses the previous 30 observed days. Later test windows may include earlier observed 2017 temperatures; the model does not receive the current target.')
        columns = {'true_temp':'Observed',f'lstm_pred_{suffix}':'LSTM',f'tcn_pred_{suffix}':'TCN'}
    else:
        columns = {'true_temp':'Observed','lstm_pred_on_pred':'LSTM','tcn_pred_on_pred':'TCN'}
        st.warning('The notebook’s recursive demonstration diverges badly. It repeatedly feeds predictions into the next window, and also mixes target-scaled predictions with input-scaled values. This reproduction preserves that behavior; these curves are failure cases.')
    days = st.slider('Records to display', 10, len(weather), len(weather))
    chart = weather.iloc[:days][['date',*columns]].rename(columns=columns)
    plot(px.line(chart, x='date', y=list(columns.values()), title='Temperature forecasts', labels={'value':'Temperature (°C)','date':'Date','variable':'Series'}))
    st.subheader('Published Kaggle results versus our rerun')
    display = metrics.rename(columns={metrics.columns[0]:'Configuration'})
    display['Configuration'] = display['Configuration'].replace({
        'lstm_pred_one_ft_loss':'LSTM · temperature only',
        'tcn_pred_one_ft_loss':'TCN · temperature only',
        'lstm_pred_two_ft_loss':'LSTM · temperature + humidity',
        'tcn_pred_two_ft_loss':'TCN · temperature + humidity'})
    st.dataframe(display.round(4), hide_index=True, width='stretch')
    st.write('RMSE is the typical scale of prediction error: lower is better. MSE is in squared degrees. R² describes variance explained and is not classification accuracy. Our local best LSTM and TCN differ by only about 0.006 °C; one seeded run cannot establish a reliable winner.')
    with st.expander('How closely did we reproduce the notebook?'):
        st.write('We executed the original 74 code cells: four initial validation models, then four fresh final models trained for 47 epochs. Original architectures, splits and forecasting operations were preserved. Runtime fixes addressed callback-list syntax and indentation; saved artifacts use NumPy weights plus model JSON. Seed 42 was added for the local rerun. Different software versions and randomness mean outputs need not match published values exactly.')
        st.write('The original notebook fits the input scaler before its initial validation split, which leaks validation distribution information. The duplicate January 1 record is retained. These choices are preserved for fidelity, and should be corrected in a future controlled comparison. Initial validation RMSE is normalized; the table above reports final RMSE in °C.')
    with st.expander('Original-style output gallery'):
        figures = sorted((DATA/'climate_figures').glob('*.png'))
        selected = st.selectbox('Notebook output', figures, format_func=lambda p:p.stem)
        st.image(str(selected), width='stretch')
        st.caption('Generated during our local execution of the Kaggle notebook. Cell numbers identify the corresponding original code cell.')
    st.download_button('Download recorded predictions', (DATA/'climate_predictions.csv').read_bytes(), 'climate_predictions.csv','text/csv')

elif page == 'Understand TCN & LSTM':
    st.title('How do the models read a sequence?')
    st.write('A single frame says where a hand is. A sequence says how the hand moves. Both models combine evidence over time, but they do it differently.')
    left,right = st.columns(2)
    with left:
        st.subheader('LSTM · a running memory')
        st.code('frame 1 → memory → frame 2 → memory → … → prediction', language=None)
        st.write('An LSTM reads timesteps in order. Gates decide what to remember, forget and expose. Its hidden state summarizes the movement so far. Like reading a sentence word by word, the interpretation develops as new input arrives.')
        st.write('Our Auslan LSTM: one layer with 64 hidden units. The climate notebook stacks 64 and 128 units. State can be carried between steps in a streaming implementation.')
    with right:
        st.subheader('TCN · filters across time')
        st.code('sequence → causal filters → wider filters → prediction', language=None)
        st.write('A TCN applies learned filters across a time window. Early filters detect short movement patterns; deeper filters combine wider context. Causal filters use the present and past. Dilated filters skip positions to cover a longer history without a very large kernel.')
        st.write('Our Auslan TCN: 32 channels, four residual blocks, dilations 1, 2, 4 and 8, with two kernel-3 convolutions per block. The climate notebook uses 64 filters and layer normalization.')
    st.subheader('Explore dilation')
    dilation = st.select_slider('Dilation spacing', options=[1,2,4,8], value=4)
    step = st.slider('Current timestep', 17,30,30)
    xs = list(range(1,31)); used = [step-2*dilation,step-dilation,step]
    fig=go.Figure(go.Scatter(x=xs,y=[0]*30,mode='markers',marker=dict(size=14,color=['#ffbd59' if x in used else '#33455e' for x in xs]),hovertemplate='Timestep %{x}<extra></extra>'))
    fig.update_layout(title=f'One causal kernel-3 filter: samples {used}',xaxis_title='Timestep',yaxis=dict(visible=False),height=220)
    plot(fig)
    st.write('This illustrates one filter, not the complete network. Stacking the four two-convolution blocks gives a receptive field of 1 + 2 × (3 − 1) × (1 + 2 + 4 + 8) = 61 timesteps. With a 30-step input, the remaining history is padding.')
    st.dataframe(pd.DataFrame({'Question':['How is history combined?','Can training process timesteps in parallel?','What limits history?','Will it always be faster or more accurate?'], 'LSTM':['Recurrent hidden and cell states','Recurrence creates sequential dependencies','State capacity and optimization','No; depends on implementation and data'], 'TCN':['Stacked temporal convolution filters','Yes, within convolution layers','Receptive field and supplied window','No; depends on implementation and data']}),hide_index=True,width='stretch')

elif page == 'Our ISL project':
    st.title('Why test TCN for H.E.L.E.N?')
    st.info('Our evidence supports comparing TCN with LSTM. It does not support promising that TCN will work better on ISL.')
    st.write('ISL signs contain motion patterns: changes in hand position, orientation and finger shape. Temporal filters may learn short patterns and combine them across a wider window. Convolutions can process training timesteps in parallel, which may reduce training time on suitable hardware. A fixed causal window also gives an explicit amount of past context.')
    st.write('These are reasons to investigate TCN. Our current TCN was slower and less accurate than LSTM on the Auslan CPU baseline. Climate forecasting produced close scores and cannot establish sign-language performance. Model size, feature quality, signer variation and deployment hardware all affect the outcome.')
    st.subheader('Planned camera pipeline')
    st.code('Webcam frames → hand/body landmarks → normalized sequence\n             → TCN or LSTM → stable word decision → text/speech',language=None)
    st.write('Neither current dataset supplies ISL webcam landmarks. The glove classifier cannot be directly connected to a webcam: its eight input channels describe different measurements.')
    st.subheader('A fair next experiment')
    st.markdown('1. Select a small vocabulary and verify every sign with a competent ISL user.\n2. Collect repeated clips from consenting participants, with varied lighting and backgrounds.\n3. Keep participants separate across training, validation and test sets.\n4. Give both models the same landmark sequences, split and tuning budget; fit preprocessing only on training data.\n5. Repeat runs and compare accuracy, macro F1, confusion, model size and complete camera-to-output latency.\n6. Test the chosen model on the intended device before claiming an offline wearable solution.')
    st.subheader('A presentation explanation you can use')
    st.success('“We implemented two sequence-learning baselines and tested them on real public data. LSTM currently leads our Auslan baseline, while climate scores are close. We are investigating causal TCNs for ISL because temporal filters may capture movement efficiently. Our next milestone is signer-independent evaluation on verified ISL webcam recordings.”')
    st.caption('Speech-to-sign avatar output, continuous sentence translation, wearable deployment and environmental assistance remain future milestones.')

else:
    st.title('Sources, provenance & reproduction')
    st.markdown('- [UCI Australian Sign Language Signs dataset](https://archive.ics.uci.edu/dataset/114/australian+sign+language+signs)\n- [Ricardo Colindres: LSTM vs TCN Kaggle notebook](https://www.kaggle.com/code/ricardocolindres/lstm-vs-tcn-for-time-series-analysis-comparison)\n- [Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)\n- [TCN sequence-modeling paper: Bai, Kolter & Koltun](https://arxiv.org/abs/1803.01271)\n- [keras-tcn implementation](https://github.com/philipperemy/keras-tcn)')
    st.write('Results are bundled from the local training runs. Changing dashboard controls explores recorded outputs; it does not retrain models. No fabricated metrics, live video or generated ISL data are presented.')
    st.write('The notebook is attributed to Ricardo Colindres and licensed under Apache 2.0. See the included NOTICE and license. Dataset terms and citations remain with their source providers. Only a small set of public-recording position previews is included; full archives, student PPT, virtual environments and credentials are excluded.')
    st.download_button('Download Auslan run results',(DATA/'auslan_results.json').read_bytes(),'auslan_results.json','application/json')
    st.download_button('Download climate comparison',(DATA/'climate_comparison.csv').read_bytes(),'climate_comparison.csv','text/csv')
    st.markdown('**Run the dashboard**\n```bash\npip install -r requirements.txt\nstreamlit run app.py\n```\nFor training, see the experiment source scripts and reproduction instructions in this repository. The cloud demo uses lightweight plotting dependencies; training requires separate PyTorch or TensorFlow environments and the original datasets.')

st.divider()
st.caption('H.E.L.E.N · Research progress demonstration · Results from local runs, not a validated ISL communication system')
