const {
	GraphQLNonNull,
	GraphQLEnumType,
	GraphQLList,
	GraphQLString,
	GraphQLInputObjectType,
} = require('graphql');
const IdScalar = require('../scalars/id.scalar.js');
const UriScalar = require('../scalars/uri.scalar.js');
const CodeScalar = require('../scalars/code.scalar.js');
const UrlScalar = require('../scalars/url.scalar.js');

/**
 * @name exports
 * @summary Endpoint Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'Endpoint_Input',
	description:
		'The technical details of an endpoint that can be used for electronic services, such as for web services providing XDS.b or a REST endpoint for another FHIR server. This may include any security context information.',
	fields: () => ({
		resourceType: {
			type: new GraphQLNonNull(
				new GraphQLEnumType({
					name: 'Endpoint_Enum_input',
					values: { Endpoint: { value: 'Endpoint' } },
				}),
			),
			description: 'Type of resource',
		},
		_id: {
			type: require('./element.input.js'),
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		id: {
			type: IdScalar,
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		meta: {
			type: require('./meta.input.js'),
			description:
				'The metadata about the resource. This is content that is maintained by the infrastructure. Changes to the content might not always be associated with version changes to the resource.',
		},
		_implicitRules: {
			type: require('./element.input.js'),
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		implicitRules: {
			type: UriScalar,
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		_language: {
			type: require('./element.input.js'),
			description: 'The base language in which the resource is written.',
		},
		language: {
			type: CodeScalar,
			description: 'The base language in which the resource is written.',
		},
		text: {
			type: require('./narrative.input.js'),
			description:
				"A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human. The narrative need not encode all the structured data, but is required to contain sufficient detail to make it 'clinically safe' for a human to just read the narrative. Resource definitions may define what content should be represented in the narrative to ensure clinical safety.",
		},
		contained: {
			type: new GraphQLList(GraphQLString),
			description:
				'These resources do not have an independent existence apart from the resource that contains them - they cannot be identified independently, and nor can they have their own independent transaction scope.',
		},
		extension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the resource. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		modifierExtension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				"May be used to represent additional information that is not part of the basic definition of the resource and that modifies the understanding of the element that contains it and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and manageable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer is allowed to define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.  Modifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
		},
		identifier: {
			type: new GraphQLList(require('./identifier.input.js')),
			description:
				'Identifier for the organization that is used to identify the endpoint across multiple disparate systems.',
		},
		_status: {
			type: require('./element.input.js'),
			description: 'active | suspended | error | off | test.',
		},
		status: {
			type: new GraphQLNonNull(CodeScalar),
			description: 'active | suspended | error | off | test.',
		},
		connectionType: {
			type: new GraphQLNonNull(require('./coding.input.js')),
			description:
				'A coded value that represents the technical details of the usage of this endpoint, such as what WSDLs should be used in what way. (e.g. XDS.b/DICOM/cds-hook).',
		},
		_name: {
			type: require('./element.input.js'),
			description:
				'A friendly name that this endpoint can be referred to with.',
		},
		name: {
			type: GraphQLString,
			description:
				'A friendly name that this endpoint can be referred to with.',
		},
		managingOrganization: {
			type: GraphQLString,
			description:
				'The organization that manages this endpoint (even if technically another organization is hosting this in the cloud, it is the organization associated with the data).',
		},
		contact: {
			type: new GraphQLList(require('./contactpoint.input.js')),
			description:
				'Contact details for a human to contact about the subscription. The primary use of this for system administrator troubleshooting.',
		},
		period: {
			type: require('./period.input.js'),
			description:
				'The interval during which the endpoint is expected to be operational.',
		},
		payloadType: {
			type: new GraphQLList(
				new GraphQLNonNull(require('./codeableconcept.input.js')),
			),
			description:
				'The payload type describes the acceptable content that can be communicated on the endpoint.',
		},
		_payloadMimeType: {
			type: require('./element.input.js'),
			description:
				'The mime type to send the payload in - e.g. application/fhir+xml, application/fhir+json. If the mime type is not specified, then the sender could send any content (including no content depending on the connectionType).',
		},
		payloadMimeType: {
			type: new GraphQLList(CodeScalar),
			description:
				'The mime type to send the payload in - e.g. application/fhir+xml, application/fhir+json. If the mime type is not specified, then the sender could send any content (including no content depending on the connectionType).',
		},
		_address: {
			type: require('./element.input.js'),
			description: 'The uri that describes the actual end-point to connect to.',
		},
		address: {
			type: new GraphQLNonNull(UrlScalar),
			description: 'The uri that describes the actual end-point to connect to.',
		},
		_header: {
			type: require('./element.input.js'),
			description:
				'Additional headers / information to send as part of the notification.',
		},
		header: {
			type: new GraphQLList(GraphQLString),
			description:
				'Additional headers / information to send as part of the notification.',
		},
	}),
});
