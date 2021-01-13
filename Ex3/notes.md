# Notes

Please write any notes that relate to exercise 3 in here, such as:

- Usefull links,
- potential pitfalls you encountered during installation,
- important commands, parameters,... ,
- whatever you can think of.

Advantages of markdown:
It's simple and you really only need a text editor.
You can have the file open in VSCode and display the rendered result via preview on the right (might need an extension for that, not sure anymore).

Code can be inserted like this:

Bash commands

```bash
pip install numpy
```

Python

```python
print("This. Is. Python.")
```

## Topic

> Link: [Topic 3.1.2.5: Secure Multi-party computation for Image Classification](./docs/ML_WS2020_Exercise3.1.docx.pdf) (pdf: Ex3.1, page 32)

## **Task**

Utilising e.g. the library mpyc for Python (​https://github.com/lschoe/mpyc​) and the implementation of a secure computation for a binarised multi-layer perceptron, first try to recreate the results reported in Abspoel et al, “Fast Secure Comparison for Medium-Sized Integers and Its Application in Binarized Neural Networks”, i.e. train a baseline CNN to estimate a potential upper limit of achievable results, and then train the binarized network, as a simplified but still rather performant version,  in a secure way. If needed, you can use a subset of the MNIST dataset. Then, try to perform a similar evaluation on another small dataset, either already available in greyscale, or converted to greyscale, e.g. using (a subset of) the AT&T faces dataset. Specifically, evaluate the final result in terms of effectiveness, but also consider efficiency aspects,i.e. primarily runtime, but also other resource consumption.

### **Part 1**

- Install [mpyc](​https://github.com/lschoe/mpyc​)
- OR another similar library/package

### **Part 2**

- Read: [Abspoel et al, “Fast Secure Comparison for Medium-Sized Integers and Its Application in Binarized Neural Networks”](),
- Train a baseline CNN

### **Part 3**

- Train a binarized network (simplified but still performant version)

## Datasets

We should test on 2 datasets. We have to be careful to choose 2 "compatible" datasets, i.e MNIST is in greyscale --> either choose a dataset that is already in greyscale or transform the dataset into greyscale.

Suggested:

- [MNIST](https://www.openml.org/d/554) (can even use a subset if necessary)
- [AT&T faces](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/)

## References

- [mypc homepage](https://www.win.tue.nl/~berry/mpyc/)
- [Awesome Mpc: A curated list of multi party computation resources and links.](https://awesomeopensource.com/project/rdragos/awesome-mpc)
