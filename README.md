This package is designed to count animals in images and videos using object detection.

Installation
------------

To install this package, run the following command:


    python setup.py sdist bdist_wheel
    pip install dist/animalcounter-0.1.0-py3-none-any.whl

Usage
-----

Here is an example of how to use the package:


    from animal_counter import AnimalCounter

    #Create an instance of the AnimalCounter
    animal_counter = AnimalCounter()

    #Load an image
    image_path = "/path/to/image.jpg"
    image = animal_counter.load_image(image_path)

    #Get the output image with the animal count
    output_image = animal_counter.get_output_image(image)

    #Display the output image
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)

    #Load a video
    video_path = "/path/to/video.mp4"
    cap = animal_counter.load_video(video_path)

    #Loop through the frames of the video and count the animals
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            output_image = animal_counter.get_output_image(frame)
            cv2.imshow('Output Image', output_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    #Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

Dependencies
------------

This package requires the following dependencies:

-   numpy
-   torch
-   detectron2
-   matplotlib
-   opencv-python

License
-------

This package is licensed under the MIT License. See the [LICENSE](https://chat.openai.com/chat/LICENSE) file for details.
