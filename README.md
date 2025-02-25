# sentence-transformer
A sentence transformer with 3 pre-trained sentence classes: math, space, nature

## Usage

To run the application, use the `dockerrun.sh` script.<br>
This will create a docker container and publish it to the local docker registry (Docker Desktop)<br>
The port exposed is 5000.<br>
Use one of the endpoints listed below.<br>

## ENDPOINTS:
    http://localhost:5000/

    GET: /classify/<sentence>              pass in a string to be classified as either math, space, nature, or none\n
    GET: /sentiment/<sentence>             pass in a string to be classified as POSITIVE or NEGATIVE\n\n
    

    sample usage: http://localhost:5000/classify/:sentence  \n
    key: sentence  value: "The universe is expanding"