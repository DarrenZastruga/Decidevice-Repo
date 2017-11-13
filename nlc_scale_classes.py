# Copyright 2016 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The v1 Natural Language Classifier service
(https://www.ibm.com/watson/developercloud/nl-classifier.html)
"""

import json
from watson_developer_cloud.watson_developer_cloud_service import \
    WatsonDeveloperCloudService


class NaturalLanguageClassifierV1(WatsonDeveloperCloudService):
    default_url = 'https://gateway.watsonplatform.net/natural-language' \
                  '-classifier/api'

    def __init__(self, url=default_url, **kwargs):
        WatsonDeveloperCloudService.__init__(
            self, 'natural_language_classifier', url, **kwargs)

    def create(self, training_data, name=None, language='en'):
        """
        :param training_data: A csv file representing the training data
        :param name: The optional descriptive name for the classifier
        :param language: The language of the input data, i.e. 'en'
        :return: A JSON object with the classifier_id of the newly created
        classifier, still in training
        """
        params = {'language': language, 'name': name}
        return self.request(method='POST', url='/v1/classifiers',
                            accept_json=True,
                            files=[('training_metadata',
                                    ('training.json', json.dumps(params))),
                                   ('training_data', training_data)])

    def list(self):
        return self.request(method='GET', url='/v1/classifiers',
                            accept_json=True)

    def status(self, classifier_id):
        classifier_id = self.unpack_id(classifier_id, 'classifier_id')
        return self.request(method='GET',
                            url='/v1/classifiers/{0}'.format(classifier_id),
                            accept_json=True)

    def classify(self, classifier_id, text):
        classifier_id = self.unpack_id(classifier_id, 'classifier_id')
        return self.request(method='POST',
                            url='/v1/classifiers/{0}/classify'.format(
                                classifier_id), accept_json=True,
                            json={'text': text})

    def remove(self, classifier_id):
        classifier_id = self.unpack_id(classifier_id, 'classifier_id')
        return self.request(method='DELETE',
                            url='/v1/classifiers/{0}'.format(classifier_id),
                            accept_json=True)
    
    def get_nlc_response(self, input_text):
        """Call NLC with input_text and return formatted response.
        Formatted response_tuple is saved for Conversation to allow to be referenced.
        Response is then further formatted to be passed to UI.
        :param str input_text: query to be used with Watson NLC Service
        :returns: NLC response in format for Watson Conversation
        :rtype: dict
        """

        nlc_response = self.discovery_client.query(
            classifier_id=self.classifier_id,
            query_options={'query': input_text, 'count': DISCOVERY_QUERY_COUNT}
        )

        # Watson nlc assigns a confidence level to each result.
        # Based on data mix, we can assign a minimum tolerance value in an
        # attempt to filter out the "weakest" results.
        if self.nlc_score_filter and 'results' in nlc_response:
            fr = [x for x in nlc_response['results'] if 'score' in x and
                  x['score'] > self.nlc_score_filter]

            nlc_response['matching_results'] = len(fr)
            nlc_response['results'] = fr

        response = self.format_nlc_response(nlc_response,
                                                  self.dnlc_data_source)
        self.response_tuple = response

        fmt = "{cart_number}) {name}\n{image}"
        formatted_response = "\n".join(fmt.format(**item) for item in response)
        return {'nlc_result': formatted_response}
     def handle_nlc_query(self):
        """Take query string from Watson Context and send to the Natural Language Classifier (NLC) service.
       NLC response will be merged into context in order to allow it to
        be returned to Watson. In the case where there is no NLC client,
        a fake response will be returned, for testing purposes.
        :returns: False indicating no need for UI input, just return to Watson
        :rtype: Bool
        """
        query_string = self.context['nlc_string']
        if self.nlc_client:
            try:
                response = self.get_nlc_response(query_string)
            except Exception as e:
                response = {'nlc_result': repr(e)}
        else:
            response = self.get_fake_nlc_response()

        self.context = self.context_merge(self.context, response)
        LOG.debug("watson_nlc:\n{}\ncontext:\n{}".format(response,
                                                               self.context))

        # no need for user input, return to Watson Dialogue
        return False

    def get_watson_response(self, message):
        """Sends text and context to Watson and gets reply.
        Message input is text, self.context is also added and sent to Watson.
        :param str message: text to send to Watson
        :returns: json dict from Watson
        :rtype: dict
        """
        response = self.conversation_client.message(
            workspace_id=self.workspace_id,
            message_input={'text': message},
            context=self.context)
        return response
