const {
	GraphQLString,
	GraphQLList,
	GraphQLInputObjectType,
} = require('graphql');
const CodeScalar = require('../scalars/code.scalar.js');
const Base64BinaryScalar = require('../scalars/base64binary.scalar.js');
const UrlScalar = require('../scalars/url.scalar.js');
const UnsignedIntScalar = require('../scalars/unsignedint.scalar.js');
const DateTimeScalar = require('../scalars/datetime.scalar.js');

/**
 * @name exports
 * @summary Attachment Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'Attachment_Input',
	description:
		'Base StructureDefinition for Attachment Type: For referring to data content defined in other formats.',
	fields: () => ({
		_id: {
			type: require('./element.input.js'),
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		id: {
			type: GraphQLString,
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		extension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the element. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		_contentType: {
			type: require('./element.input.js'),
			description:
				'Identifies the type of the data in the attachment and allows a method to be chosen to interpret or render the data. Includes mime type parameters such as charset where appropriate.',
		},
		contentType: {
			type: CodeScalar,
			description:
				'Identifies the type of the data in the attachment and allows a method to be chosen to interpret or render the data. Includes mime type parameters such as charset where appropriate.',
		},
		_language: {
			type: require('./element.input.js'),
			description:
				'The human language of the content. The value can be any valid value according to BCP 47.',
		},
		language: {
			type: CodeScalar,
			description:
				'The human language of the content. The value can be any valid value according to BCP 47.',
		},
		_data: {
			type: require('./element.input.js'),
			description:
				'The actual data of the attachment - a sequence of bytes, base64 encoded.',
		},
		data: {
			type: Base64BinaryScalar,
			description:
				'The actual data of the attachment - a sequence of bytes, base64 encoded.',
		},
		_url: {
			type: require('./element.input.js'),
			description: 'A location where the data can be accessed.',
		},
		url: {
			type: UrlScalar,
			description: 'A location where the data can be accessed.',
		},
		_size: {
			type: require('./element.input.js'),
			description:
				'The number of bytes of data that make up this attachment (before base64 encoding, if that is done).',
		},
		size: {
			type: UnsignedIntScalar,
			description:
				'The number of bytes of data that make up this attachment (before base64 encoding, if that is done).',
		},
		_hash: {
			type: require('./element.input.js'),
			description:
				'The calculated hash of the data using SHA-1. Represented using base64.',
		},
		hash: {
			type: Base64BinaryScalar,
			description:
				'The calculated hash of the data using SHA-1. Represented using base64.',
		},
		_title: {
			type: require('./element.input.js'),
			description: 'A label or set of text to display in place of the data.',
		},
		title: {
			type: GraphQLString,
			description: 'A label or set of text to display in place of the data.',
		},
		_creation: {
			type: require('./element.input.js'),
			description: 'The date that the attachment was first created.',
		},
		creation: {
			type: DateTimeScalar,
			description: 'The date that the attachment was first created.',
		},
	}),
});
