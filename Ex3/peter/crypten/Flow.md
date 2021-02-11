# Execution flow

## Scenarios

*We will assume 2 participants, Alice and Bob, for ease of explanation.*

Different MPC scenarios as it relates to ML:

1. Participants have different data of same shape + train same model:
    - Alice has 1000 pictures of digits
    - Bob has another, different set of 5000 pictures of digits
    - Alice and Bob want to train a model with 6000 pictures in total, without revealing their pictures to each other
2. Participants hold different pieces of the same puzzle/data:
    a.  
        - Alice has the 6000 pictures of digits, but doesn't know which is which
        - Bob does not have the pictures, but knows what the pictures each show
            - This setup is a bit... unrealistic for the use case maybe
        - Nevertheless, they want to combine their data and train a model
    b.  
        - Alice has the top half of 3000 pictures of digits
        - Bob has the bottom half of the same 3000 pictures of digits
            - Again, this is a bit weird...
        - Nevertheless, they want to combine their data and train a model
3. One participant owns data, the other one owns the model:
    - Alice has all the data
    - Bob wants to train his model on Alice's data
    - Bob later wants to share his model:
        - the encrypted model (readable by Bob) is fed with
        - encrypted data (readable by Alice),
        - but the output should be visible (decryptable) by both.
        
Obviously, combinations can occur especially once more participants are involved.

## Which scenarios do we want to benchmark?

My opinion: Only 1 and 3 are relevant for the image classification, so I think we can focus on that.

## Benchmark 
    
1. We seperate the data into batches as we want
    - Split into train, validation and test set? 80 : 10 : 10 %
        - Both the normal and encrypted model is trained on 
    - How to get the data to every host?
    - Maybe host a seperate repository to clone from?
2. Only the relevant data is loaded by each process/participant
3. Model is trained
    - Benchmark time needed for training
    - Log 
    - Ideally, the same script could be used for single participant (=normal) training
4. Evaluate model
    - Compare performance on train, validation and test set (maybe only train+test?) to normal model
        - Accuracy
        - F1 = 2 * recall * precision / (recall + precision)
        - Training time
        - Runtime/inference time
        - Efficiency = accuracy / time
        - Memory consumption?
        - Size of saved model on disk?
        
Encrypted (binarized) vs normal

P1 - P2 - P3
    
    