/**
 * Returns a Datetime string. Example: 2021-03-24 06:28:02
 * @returns Date
 */
const getCurrentDate = () => new Date().toISOString().replace(/T/, ' ').replace(/\..+/, '');

module.exports = { getCurrentDate };
