# Protein Folding 2D HP-Model Simulation

Protein plays a big role for every living organismm, it keeps us healthy by performs it functionality such as transporting oxygen, part of imune system, muscle development, etc. These amazing functionality only works if those protein 'fold' into a stable(native)-state where it has a minimum 'free-energy' value. Predicting the minimum 'free-energy', helps bioinformatics, biotechnology in terms of developing a novel drug, vaccine, and many more. This project helps to simulate the folding process in 2D HP-model which is also well known as NP-complete problem.

## Getting Started

If you ever play around any of **OpenAI gym** environment, this project follows the **OpenAI gym** funtions behaviour.

Please run the code **create_background.py** inside '**protein_folding\Code**' folder. 
It will generate an initial background '**.npy**' file for visualization purpose.

![create_background](https://user-images.githubusercontent.com/58515206/84480578-f60c8200-acbe-11ea-9cc2-ad220a287f38.PNG)

### Prerequisites

* numpy               >=   1.18.2
* matplotlib          >=   3.2.1
* opencv-python       >=   4.2.0.34
* Pillow              >=   7.1.2

### Installation
Open Terminal and install the above prerequisites libraries

Numpy
```
pip install numpy
```
Matplotlib
```
pip install matplotlib
```
OpenCv
```
pip install opencv-python
```
Pillow
```
pip install pillow
```

## How it works?

HP-model looks like a simple **board game**, since 20 different amino acid in protein are classified into 2 amino acid:
*	**‘H’ = Hydrophobic**  (Black Circle)
*	**‘P’ = Hydrophillic** (White Circle)

Given a sequence of amino **['H', 'P']**, the agent task is to place each amino in the sequence into 2D space. Note that, the next amino should be place side by side up, left, right, down from the previous amino. Repeat this process until all amino in the sequence has been placed.

**Example** :

         UP                  Left                 Right                Down

<img src="https://user-images.githubusercontent.com/58515206/84485971-eee97200-acc6-11ea-9d24-e0ed2f09b990.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84485971-eee97200-acc6-11ea-9d24-e0ed2f09b990.PNG" width="150" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84487608-32dd7680-acc9-11ea-98c1-705e57684f7b.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84488014-cadb6000-acc9-11ea-9a63-22b659dbe3cb.PNG" width="150" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84488014-cadb6000-acc9-11ea-9a63-22b659dbe3cb.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84487959-b7c89000-acc9-11ea-9a2f-259b96c88e7a.PNG" width="150" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84488248-1c83ea80-acca-11ea-937e-ad5463365637.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84488248-1c83ea80-acca-11ea-937e-ad5463365637.PNG" width="150" height="200" />

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
