from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    out = {}
    if request.method=='POST':
        beat = request.files['beat']
        seed = request.form['seed']
        lines = generate_pipeline(beat, seed)
        meter, rhyme = analyze_meter_and_rhyme_block(lines)
        rh_high = highlight_full_rhymes("\n".join(lines))
        out = dict(lines=lines, meter=meter, rhyme=rhyme, rh_high=str(rh_high))
    return render_template('index.html', **out)

if __name__=='__main__':
    app.run(debug=True)
